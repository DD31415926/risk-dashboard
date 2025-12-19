import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 0) 配置与路径
# =========================================================
# 请确保路径正确
APP_PATH  = "application_record.csv"
CRED_PATH = "credit_record.csv"

TEST_SIZE = 0.3
RANDOM_STATE = 42

# Basic Auth (可选)
AUTH_USER = os.environ.get("DASH_AUTH_USER")
AUTH_PASS = os.environ.get("DASH_AUTH_PASS")

def add_basic_auth(server, username, password):
    if not username or not password: return
    from flask import request, Response
    @server.before_request
    def _basic_auth():
        auth = request.authorization
        if auth and auth.username == username and auth.password == password: return None
        return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="Login Required"'})

# =========================================================
# 1) 核心特征工程逻辑 (Raw Inputs -> Model Features)
# =========================================================
def engineer_features(df_in):
    """
    接收用户列出的原始字段, 转换为模型所需的工程特征。
    """
    df = df_in.copy()
    
    # --- 1. 处理 MONTHS_BALANCE ---
    # 在原始 credit 表中，MONTHS_BALANCE 是 0, -1, -2...
    # 在预测器输入中，用户输入历史长度（如 -12 或 12）
    # 我们将其转换为 Account_Age_Months (正数)
    if "MONTHS_BALANCE" in df.columns:
        # 如果输入是负数(如-12)，取绝对值；如果是正数(12)，直接用
        df["Account_Age_Months"] = pd.to_numeric(df["MONTHS_BALANCE"], errors='coerce').abs()
        # 简单起见，假设活跃月份等于历史长度（对于单点预测模拟足够）
        df["Active_Months"] = df["Account_Age_Months"]
    elif "Account_Age_Months" not in df.columns:
        # 兜底
        df["Account_Age_Months"] = 0
        df["Active_Months"] = 0

    # --- 2. 资产分数 (Asset Score) ---
    if "FLAG_OWN_CAR" in df.columns and "FLAG_OWN_REALTY" in df.columns:
        # 兼容 Y/N, Yes/No, 1/0
        mapper = {'Y': 1, 'N': 0, 'YES': 1, 'NO': 0, '1': 1, '0': 0}
        has_car = df["FLAG_OWN_CAR"].astype(str).str.upper().map(mapper).fillna(0)
        has_realty = df["FLAG_OWN_REALTY"].astype(str).str.upper().map(mapper).fillna(0)
        df["Asset_Score"] = has_car + has_realty
    
    # --- 3. 年龄处理 (DAYS_BIRTH -> Age_Clean) ---
    if "DAYS_BIRTH" in df.columns:
        d_birth = pd.to_numeric(df["DAYS_BIRTH"], errors='coerce')
        # 如果是原始数据(-12000)，转绝对值/365；如果是用户输入正数(32)，直接用
        df["Age_Clean"] = d_birth.apply(lambda x: abs(x)/365.25 if x < 0 else x)
    
    # --- 4. 工作年限 (DAYS_EMPLOYED -> Clean) ---
    if "DAYS_EMPLOYED" in df.columns:
        d_emp = pd.to_numeric(df["DAYS_EMPLOYED"], errors='coerce')
        # 365243 是原始数据中的异常值
        df["DAYS_EMPLOYED_CLEAN"] = d_emp.apply(lambda x: 0 if x > 200000 else abs(x)/365.25)

    # --- 5. 家庭收入效率 ---
    # 需要: AMT_INCOME_TOTAL, CNT_FAM_MEMBERS, CNT_CHILDREN
    cols = ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "CNT_CHILDREN"]
    if set(cols).issubset(df.columns):
        inc = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors='coerce').fillna(0)
        fam = pd.to_numeric(df["CNT_FAM_MEMBERS"], errors='coerce').fillna(1)
        child = pd.to_numeric(df["CNT_CHILDREN"], errors='coerce').fillna(0)
        denom = (fam - child).clip(lower=1)
        df["Adult_Income_Efficiency"] = inc / denom

    # --- 6. Flag 处理 (Y/N -> 0/1) ---
    # 这一步是为了确保 Flag 列如果是 Object 类型也能被后续 Pipeline 识别为 Binary
    # 但实际上我们的 Pipeline 会对 Categorical 做 OneHot，对 Numeric 做 Poly
    # 这里不需要额外操作，交给 Pipeline 即可。
    
    return df

# =========================================================
# 2) 数据加载与预处理 (训练阶段)
# =========================================================
print("Loading and Merging Data...")
app_df = pd.read_csv(APP_PATH)
cred_df = pd.read_csv(CRED_PATH)

# --- Labeling (Vintage Analysis) ---
WINDOW = 12
BAD_SET = {"1", "2", "3", "4", "5"}

# 筛选窗口数据
w = cred_df[(cred_df["MONTHS_BALANCE"] >= -WINDOW) & (cred_df["MONTHS_BALANCE"] <= 0)].copy()

# 计算聚合特征（用于Label）
grp = w.groupby("ID")
months_in_window = grp["MONTHS_BALANCE"].nunique()
ever_bad = grp["STATUS"].apply(lambda s: s.isin(BAD_SET).any())
ever_c   = grp["STATUS"].apply(lambda s: (s == "C").any())

lab = pd.DataFrame({
    "ID": months_in_window.index,
    "months_in_window": months_in_window.values,
    "ever_bad": ever_bad.reindex(months_in_window.index).values,
    "ever_c": ever_c.reindex(months_in_window.index).values,
})

# 定义 Label
lab["label"] = np.nan
full_cond = lab["months_in_window"] >= (WINDOW + 1)
lab.loc[full_cond & (lab["ever_bad"]), "label"] = 1
lab.loc[full_cond & (~lab["ever_bad"]) & (lab["ever_c"]), "label"] = 0
labels = lab[["ID", "label"]].dropna().copy().astype(int)

# --- Merge ---
merged_df = app_df.merge(labels, on="ID", how="inner")

# --- 准备 MONTHS_BALANCE 特征用于训练 ---
# 注意：原始 application_record 没有 MONTHS_BALANCE，我们需要从 credit_record 算出来合并进去
# 这样才能保证 engineer_features 函数在训练和预测时输入一致
acct_stats = cred_df.groupby("ID")["MONTHS_BALANCE"].min().reset_index()
acct_stats.rename(columns={"MONTHS_BALANCE": "MONTHS_BALANCE"}, inplace=True) 
# 这里 MONTHS_BALANCE 是如 -12, -20 等最小值

merged_df = merged_df.merge(acct_stats, on="ID", how="left")

# --- 应用特征工程 ---
# 这步生成 Age_Clean, Asset_Score 等，但保留原始列
model_ready_df = engineer_features(merged_df)

# 定义要扔掉的列：
# 1. ID, label 不进入特征
# 2. 原始的 DAYS_BIRTH 等如果不想让模型直接用（防止共线性），可以扔掉。
#    但为了保险起见，我们可以保留原始 Numeric 列让 Pipeline 选择，或者只留 Engineered 列。
#    这里选择：保留 Engineered 列 + 必须的原始 Categorical 列
DROP_COLS = ["ID", "label"] 
# 注意：我们保留 DAYS_BIRTH 等原始列在 DataFrame 中，但在 ColumnTransformer 中决定用不用

y = model_ready_df["label"]
X = model_ready_df.drop(columns=DROP_COLS, errors="ignore")

# =========================================================
# 3) 训练流水线
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 自动识别列类型 (基于 model_ready_df)
# 我们需要确保只使用有效的特征列
# 排除掉用来生成的原始 Numeric 列（如 DAYS_BIRTH），只用 Age_Clean？
# 或者全放进去让 Lasso/RF 自己选。这里为了效果，我们主要使用 Engineered Numeric + Raw Categorical

# 筛选 Numeric: 包含 engineered (Age_Clean) 和 raw (CNT_CHILDREN)
# 排除 list
EXCLUDE_NUM = ["ID", "label"]
num_all = [c for c in X.select_dtypes(include=[np.number]).columns if c not in EXCLUDE_NUM]

# 筛选 Categorical
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 细分 Numeric
num_bin = [c for c in num_all if X[c].nunique() <= 2]
num_poly_base = [c for c in num_all if X[c].nunique() > 2]

print(f"Features: {len(cat_cols)} Categorical, {len(num_all)} Numerical")

preprocess = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
        
        ("num_bin", Pipeline([
            ("imp", SimpleImputer(strategy="median"))
        ]), num_bin),
        
        ("num_poly", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False))
        ]), num_poly_base),
    ],
    remainder="drop",
    verbose_feature_names_out=False
).set_output(transform="pandas")

print("Transforming and Filtering Features...")
Z_train = preprocess.fit_transform(X_train)
Z_test = preprocess.transform(X_test)

# 去除交互项 (只保留 x^2, 去掉 x*y)
poly_cols_all = [c for c in Z_train.columns if any(base in c for base in num_poly_base)]
poly_pure_cols = [c for c in poly_cols_all if " " not in c] 
non_poly_cols = [c for c in Z_train.columns if c not in poly_cols_all]

Z_train_sel = pd.concat([Z_train[non_poly_cols], Z_train[poly_pure_cols]], axis=1)
Z_test_sel = pd.concat([Z_test[non_poly_cols], Z_test[poly_pure_cols]], axis=1)

# T-test 筛选
def ttest_pvalue(x, y):
    try: return stats.ttest_ind(x[y==0], x[y==1], equal_var=False, nan_policy="omit")[1]
    except: return np.nan

pvals = [(c, ttest_pvalue(Z_train_sel[c], y_train)) for c in poly_pure_cols]
sig_poly = [c for c, p in pvals if p < 0.05]
# 兜底
if not sig_poly: sig_poly = [c for c, p in sorted(pvals, key=lambda x: x[1] if pd.notna(x[1]) else 1)[:20]]

final_cols = non_poly_cols + sig_poly
X_train_final = Z_train_sel[final_cols]
X_test_final = Z_test_sel[final_cols]

print(f"Final Model Feature Count: {len(final_cols)}")

# --- Models ---
# 1. Logistic Regression
scaler = StandardScaler(with_mean=False)
Xtr_sc = scaler.fit_transform(X_train_final)
Xte_sc = scaler.transform(X_test_final)
lr_model = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
lr_model.fit(Xtr_sc, y_train)
p_lr = lr_model.predict_proba(Xte_sc)[:, 1]
auc_lr = roc_auc_score(y_test, p_lr)
ap_lr = average_precision_score(y_test, p_lr)

# 2. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=5, 
    random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
)
rf_model.fit(X_train_final, y_train)
p_rf = rf_model.predict_proba(X_test_final)[:, 1]
auc_rf = roc_auc_score(y_test, p_rf)
ap_rf = average_precision_score(y_test, p_rf)

store = {
    "Logistic Regression": {"y": y_test.values, "p": p_lr, "auc": auc_lr, "ap": ap_lr},
    "Random Forest":       {"y": y_test.values, "p": p_rf, "auc": auc_rf, "ap": ap_rf}
}

rf_imp = pd.DataFrame({"Feature": X_train_final.columns, "Importance": rf_model.feature_importances_}).sort_values("Importance", ascending=False)


# =========================================================
# 4) Dashboard 构建 (严格使用您指定的原始字段)
# =========================================================
app = dash.Dash(__name__)
server = app.server
add_basic_auth(server, AUTH_USER, AUTH_PASS)
app.title = "Raw Data Risk Predictor"

# 定义您想要的原始字段列表
RAW_CAT_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", 
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", 
    "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"
]

RAW_NUM_COLS = [
    "AMT_INCOME_TOTAL", "CNT_CHILDREN", "CNT_FAM_MEMBERS", 
    "DAYS_BIRTH", "DAYS_EMPLOYED", "MONTHS_BALANCE"
]

# 提取 Categorical 的选项
options_map = {}
for c in RAW_CAT_COLS:
    if c in app_df.columns:
        opts = sorted([str(x) for x in app_df[c].dropna().unique()])
        options_map[c] = opts
    else:
        # Fallback (Flags usually 0/1)
        options_map[c] = ["0", "1"]

# 手动修正 Flag 的显示，使其更易读
for flag in ["FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
    if flag in options_map:
        # 确保包含 0 和 1 (或者 Y/N)
        curr = set(options_map[flag])
        if "0" not in curr and "1" not in curr:
             options_map[flag] = ["0", "1"] # 默认为0/1如果找不到
        else:
             options_map[flag] = sorted(list(curr))

def generate_raw_inputs():
    inputs = []
    
    # 1. 类别型字段 (Categorical + Flags)
    inputs.append(html.H4("Categorical & Flags", style={"marginTop": "10px", "borderBottom": "1px solid #eee"}))
    row = []
    for c in RAW_CAT_COLS:
        comp = html.Div([
            html.Label(c, style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(
                id={'type': 'in', 'index': c}, 
                options=[{'label': i, 'value': i} for i in options_map.get(c, [])], 
                placeholder=f"Select {c}",
                style={"fontSize": "12px"}
            )
        ], style={"width": "23%", "display": "inline-block", "marginRight": "2%", "marginBottom": "15px"})
        row.append(comp)
    inputs.append(html.Div(row))

    # 2. 数值型字段 (Numerical)
    inputs.append(html.H4("Numerical Features", style={"marginTop": "10px", "borderBottom": "1px solid #eee"}))
    row = []
    
    # 辅助文本字典
    help_text = {
        "DAYS_BIRTH": "e.g. -12000 (days) or 33 (years)",
        "DAYS_EMPLOYED": "e.g. -2000 (days) or 5 (years)",
        "MONTHS_BALANCE": "e.g. -12 (account history length)"
    }
    
    for c in RAW_NUM_COLS:
        placeholder = help_text.get(c, "0")
        comp = html.Div([
            html.Label(c, style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id={'type': 'in', 'index': c}, 
                type="number", 
                placeholder=placeholder,
                style={"width": "100%", "padding": "6px", "fontSize": "12px"}
            )
        ], style={"width": "23%", "display": "inline-block", "marginRight": "2%", "marginBottom": "15px"})
        row.append(comp)
    inputs.append(html.Div(row))
    
    return inputs

app.layout = html.Div([
    html.H2("Credit Risk Predictor: Raw Data Input"),
    
    # 控制面板
    html.Div([
        html.Div([
            html.Label("Select Model:"),
            dcc.Dropdown(id="model_sel", options=[{"label": k, "value": k} for k in store.keys()], value="Random Forest", clearable=False)
        ], style={"width": "30%", "display": "inline-block"}),
        
        html.Div([
            html.Label("Threshold Slider:"),
            dcc.Slider(id="threshold", min=0, max=1, step=0.01, value=0.5, marks={0: "0", 0.5: "0.5", 1: "1"})
        ], style={"width": "60%", "display": "inline-block", "marginLeft": "5%", "verticalAlign": "top"})
    ], style={"padding": "15px", "backgroundColor": "#f8f9fa", "marginBottom": "20px"}),

    dcc.Tabs([
        # Tab 1: 评估 (包含所有曲线)
        dcc.Tab(label="Model Evaluation", children=[
            html.Div([
                # 左侧：混淆矩阵 + 指标文本
                html.Div([
                    dcc.Graph(id="conf_matrix"),
                    html.Div(id="metrics_text", style={"padding": "15px", "border": "1px solid #ddd", "marginTop": "10px", "backgroundColor": "white"})
                ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top"}),

                # 右侧：ROC + PR 曲线
                html.Div([
                    dcc.Graph(id="roc_curve"),
                    dcc.Graph(id="pr_curve")
                ], style={"width": "58%", "display": "inline-block", "marginLeft": "2%"})
            ], style={"padding": "20px"}),
            
            # 只有 RF 显示 Importance
            html.Div(id="imp_wrapper", children=[
                html.H4("Feature Importance (Top 15)"),
                dash_table.DataTable(
                    data=rf_imp.head(15).to_dict("records"),
                    columns=[{"name": i, "id": i} for i in rf_imp.columns],
                    page_size=10,
                    style_cell={"textAlign": "left"}
                )
            ], style={"padding": "20px"})
        ]),

        # Tab 2: 预测器 (Raw Input)
        dcc.Tab(label="Prediction Simulator", children=[
            html.Div([
                html.P("Input data directly from raw CSV format columns.", style={"color": "#666"}),
                html.Div(generate_raw_inputs()),
                html.Button("Predict Risk", id="btn_run", n_clicks=0, style={"marginTop": "20px", "fontSize": "16px", "padding": "10px 20px", "backgroundColor": "#007bff", "color": "white"}),
                html.Hr(),
                html.Div(id="pred_result", style={"fontSize": "22px", "fontWeight": "bold", "padding": "10px"})
            ], style={"padding": "30px", "maxWidth": "1100px", "margin": "0 auto"})
        ])
    ])
], style={"maxWidth": "1300px", "margin": "0 auto", "fontFamily": "Segoe UI, Arial, sans-serif"})

# =========================================================
# 5) Callbacks
# =========================================================

def compute_metrics(y_true, y_proba, thr):
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / len(y_true)
    return tn, fp, fn, tp, prec, rec, f1, acc

@app.callback(
    [Output("conf_matrix", "figure"),
     Output("metrics_text", "children"),
     Output("roc_curve", "figure"),
     Output("pr_curve", "figure"),
     Output("imp_wrapper", "style")],
    [Input("model_sel", "value"),
     Input("threshold", "value")]
)
def update_eval(model_name, thr):
    d = store[model_name]
    thr = float(thr)
    
    tn, fp, fn, tp, prec, rec, f1, acc = compute_metrics(d["y"], d["p"], thr)
    
    # 1. CM
    cm = [[tn, fp], [fn, tp]]
    fig_cm = px.imshow(cm, text_auto=True, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                       title=f"Confusion Matrix ({model_name})", color_continuous_scale="Blues")
    fig_cm.update_layout(coloraxis_showscale=False)
    
    # 2. Text
    txt = [
        html.Div([html.Span("Accuracy: ", style={"fontWeight": "bold"}), f"{acc:.2%}"]),
        html.Div([html.Span("Precision: ", style={"fontWeight": "bold"}), f"{prec:.2%}"]),
        html.Div([html.Span("Recall: ", style={"fontWeight": "bold"}), f"{rec:.2%}"]),
        html.Div([html.Span("F1 Score: ", style={"fontWeight": "bold"}), f"{f1:.3f}"]),
        html.Br(),
        html.Div([html.Span("ROC AUC: ", style={"fontWeight": "bold"}), f"{d['auc']:.3f}"]),
        html.Div([html.Span("Avg Precision: ", style={"fontWeight": "bold"}), f"{d['ap']:.3f}"])
    ]
    
    # 3. ROC
    fpr, tpr, _ = roc_curve(d["y"], d["p"])
    fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, name="ROC", fill='tozeroy'))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig_roc.update_layout(title="ROC Curve", height=300, margin=dict(t=30, b=20, l=40, r=20))
    
    # 4. PR Curve
    p_arr, r_arr, _ = precision_recall_curve(d["y"], d["p"])
    fig_pr = go.Figure(go.Scatter(x=r_arr, y=p_arr, name="PR", fill='tozeroy', line=dict(color="orange")))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision",
                         height=300, margin=dict(t=30, b=20, l=40, r=20))
    
    # 5. Imp visibility
    style = {"display": "block", "padding": "20px"} if model_name == "Random Forest" else {"display": "none"}
    
    return fig_cm, txt, fig_roc, fig_pr, style

@app.callback(
    Output("pred_result", "children"),
    Input("btn_run", "n_clicks"),
    State({'type': 'in', 'index': ALL}, 'value'),
    State({'type': 'in', 'index': ALL}, 'id'),
    State("model_sel", "value"),
    State("threshold", "value"),
    prevent_initial_call=True
)
def run_prediction(n, values, ids, model_name, thr):
    # 1. 收集原始输入
    raw_data = {item['index']: val for val, item in zip(values, ids)}
    df_raw = pd.DataFrame([raw_data])
    
    if df_raw.isnull().all().all():
        return "Please input data values."
    
    try:
        # 2. 补全缺失的列 (防止 Pipeline 报错)
        # Dashboard 中可能有漏填的，用 None 填充，SimpleImputer 会处理
        for col in RAW_CAT_COLS + RAW_NUM_COLS:
            if col not in df_raw.columns:
                df_raw[col] = np.nan
        
        # 3. 特征工程 (Raw -> Features)
        df_eng = engineer_features(df_raw)
        
        # 4. Transform
        Z_in = preprocess.transform(df_eng)
        
        # 5. Filter Columns
        Z_sel = pd.concat([Z_in[non_poly_cols], Z_in[poly_pure_cols]], axis=1)
        X_final = Z_sel[final_cols]
        
        # 6. Predict
        if model_name == "Logistic Regression":
            X_sc = scaler.transform(X_final)
            prob = lr_model.predict_proba(X_sc)[0, 1]
        else:
            prob = rf_model.predict_proba(X_final)[0, 1]
            
        thr = float(thr)
        is_risk = prob >= thr
        lbl = "HIGH RISK (Denied)" if is_risk else "LOW RISK (Approved)"
        color = "#dc3545" if is_risk else "#28a745"
        
        return html.Div([
            html.Div(f"Model: {model_name} | Threshold: {thr}"),
            html.Div(f"Probability: {prob:.2%}"),
            html.Div(lbl, style={"color": color, "fontWeight": "bold", "fontSize": "28px", "marginTop": "10px"})
        ])
        
    except Exception as e:
        import traceback
        return html.Div([
            html.Div(f"Error: {str(e)}", style={"color": "red"}),
            html.Pre(traceback.format_exc(), style={"fontSize": "10px"})
        ])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run_server(host="0.0.0.0", port=port, debug=True)

