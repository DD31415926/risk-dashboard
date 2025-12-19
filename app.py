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
APP_PATH  = "/home/yaolushen/fall_242A/242A_datasets/application_record.csv"
CRED_PATH = "/home/yaolushen/fall_242A/242A_datasets/credit_record.csv"

TEST_SIZE = 0.3
RANDOM_STATE = 42

# Basic Auth
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
# 1) 核心特征工程逻辑 (这也是连接原始输入和模型的桥梁)
# =========================================================
def engineer_features(df_in):
    """
    接收包含原始列(AMT_INCOME_TOTAL, FLAG_OWN_CAR等)的DataFrame,
    返回包含模型所需特征(Asset_Score, Age_Clean等)的DataFrame。
    """
    df = df_in.copy()
    
    # 1. 资产分数 (Asset Score)
    # 兼容处理：如果是预测输入，可能是字面量；如果是原始读取，可能是Y/N
    if "FLAG_OWN_CAR" in df.columns and "FLAG_OWN_REALTY" in df.columns:
        has_car = df["FLAG_OWN_CAR"].astype(str).str.upper().replace({'Y': 1, 'N': 0, 'YES': 1, 'NO': 0})
        has_realty = df["FLAG_OWN_REALTY"].astype(str).str.upper().replace({'Y': 1, 'N': 0, 'YES': 1, 'NO': 0})
        # 强制转为numeric，无法转换的设为0
        has_car = pd.to_numeric(has_car, errors='coerce').fillna(0)
        has_realty = pd.to_numeric(has_realty, errors='coerce').fillna(0)
        df["Asset_Score"] = has_car + has_realty
    
    # 2. 年龄处理 (Age Clean)
    # 如果输入的是原始 DAYS_BIRTH (负数天数)
    if "DAYS_BIRTH" in df.columns:
        # 确保是数值
        d_birth = pd.to_numeric(df["DAYS_BIRTH"], errors='coerce')
        # 如果是负数（原始数据），转为正年；如果是正数（可能是用户直接填了年龄），保持不变
        df["Age_Clean"] = d_birth.apply(lambda x: abs(x)/365.25 if x < 0 else x)
    
    # 3. 工作年限 (Days Employed)
    if "DAYS_EMPLOYED" in df.columns:
        d_emp = pd.to_numeric(df["DAYS_EMPLOYED"], errors='coerce')
        # 365243 是原始数据中的异常值代表退休/无工作
        df["DAYS_EMPLOYED_CLEAN"] = d_emp.apply(lambda x: 0 if x > 0 else abs(x)/365.25)

    # 4. 家庭收入效率 (Income Efficiency)
    cols = ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "CNT_CHILDREN"]
    if set(cols).issubset(df.columns):
        inc = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors='coerce')
        fam = pd.to_numeric(df["CNT_FAM_MEMBERS"], errors='coerce')
        child = pd.to_numeric(df["CNT_CHILDREN"], errors='coerce')
        denom = (fam - child).clip(lower=1) # 避免除以0
        df["Adult_Income_Efficiency"] = inc / denom

    return df

# =========================================================
# 2) 数据加载与预处理
# =========================================================
print("Loading Data...")
app_df = pd.read_csv(APP_PATH)
cred_df = pd.read_csv(CRED_PATH)

# --- 2.1 标签定义 (Labeling) ---
WINDOW = 12
BAD_SET = {"1", "2", "3", "4", "5"}

w = cred_df[(cred_df["MONTHS_BALANCE"] >= -WINDOW) & (cred_df["MONTHS_BALANCE"] <= 0)].copy()
months_in_window = w.groupby("ID")["MONTHS_BALANCE"].nunique()
ever_bad = w.groupby("ID")["STATUS"].apply(lambda s: s.isin(BAD_SET).any())
ever_c   = w.groupby("ID")["STATUS"].apply(lambda s: (s == "C").any())

lab = pd.DataFrame({
    "ID": months_in_window.index,
    "months_in_window": months_in_window.values,
    "ever_bad": ever_bad.reindex(months_in_window.index).values,
    "ever_c": ever_c.reindex(months_in_window.index).values,
})
lab["label"] = np.nan
full = lab["months_in_window"] >= (WINDOW + 1)
lab.loc[full & (lab["ever_bad"]), "label"] = 1
lab.loc[full & (~lab["ever_bad"]) & (lab["ever_c"]), "label"] = 0
labels = lab[["ID", "label"]].dropna().copy().astype(int)

# --- 2.2 合并数据 ---
# 只做 Inner Join，防止无标签数据混入
merged_df = app_df.merge(labels, on="ID", how="inner")

# --- 2.3 补充 Credit 表中的特征 (Account Age) ---
# 这些特征必须保留，因为它们对模型很重要。
# 在预测器中，我们需要让用户手动输入这些值（模拟查询信用局）
acct_stats = cred_df.groupby("ID")["MONTHS_BALANCE"].agg(["min", "count"]).reset_index()
acct_stats["Account_Age_Months"] = acct_stats["min"].abs()
acct_stats["Active_Months"] = acct_stats["count"]
merged_df = merged_df.merge(acct_stats[["ID", "Account_Age_Months", "Active_Months"]], on="ID", how="left")

# --- 2.4 应用特征工程 ---
# 这里生成模型真正用到的列 (Age_Clean, Asset_Score 等)
model_ready_df = engineer_features(merged_df)

# 定义需要丢弃的原始列（因为已经转化为了新特征，避免多重共线性）
# 注意：我们保留原始列用于 Dashboard 的 Dropdown 选项读取，但在进入 X 之前丢弃
DROP_FOR_TRAIN = [
    "ID", "label", 
    "DAYS_BIRTH", "DAYS_EMPLOYED", 
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", 
    "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"
]

y = model_ready_df["label"]
X = model_ready_df.drop(columns=[c for c in DROP_FOR_TRAIN if c in model_ready_df.columns], errors="ignore")

# =========================================================
# 3) 模型训练流水线
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 自动识别列类型
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_all = X.select_dtypes(include=[np.number]).columns.tolist()
num_bin = [c for c in num_all if X[c].nunique() <= 2]
num_poly_base = [c for c in num_all if X[c].nunique() > 2]

# 预处理 Pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ("num_bin", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_bin),
        ("num_poly", Pipeline([("imp", SimpleImputer(strategy="median")), ("poly", PolynomialFeatures(degree=2, include_bias=False))]), num_poly_base),
    ],
    remainder="drop",
    verbose_feature_names_out=False
).set_output(transform="pandas")

print("Transforming features...")
Z_train = preprocess.fit_transform(X_train)
Z_test = preprocess.transform(X_test)

# 去除交互项 + T-test 筛选
poly_cols_all = [c for c in Z_train.columns if any(base in c for base in num_poly_base)]
poly_pure_cols = [c for c in poly_cols_all if " " not in c] # 只留 x^2, 去掉 x y
non_poly_cols = [c for c in Z_train.columns if c not in poly_cols_all]

Z_train_sel = pd.concat([Z_train[non_poly_cols], Z_train[poly_pure_cols]], axis=1)
Z_test_sel = pd.concat([Z_test[non_poly_cols], Z_test[poly_pure_cols]], axis=1)

# T-test
def ttest_pvalue(x, y):
    try: return stats.ttest_ind(x[y==0], x[y==1], equal_var=False, nan_policy="omit")[1]
    except: return np.nan

pvals = [(c, ttest_pvalue(Z_train_sel[c], y_train)) for c in poly_pure_cols]
sig_poly = [c for c, p in pvals if p < 0.05]
if not sig_poly: sig_poly = [c for c, p in sorted(pvals, key=lambda x: x[1] if pd.notna(x[1]) else 1)[:20]]

final_cols = non_poly_cols + sig_poly
X_train_final = Z_train_sel[final_cols]
X_test_final = Z_test_sel[final_cols]

print(f"Training Models with {len(final_cols)} features...")

# --- 训练 Logistic Regression ---
scaler = StandardScaler(with_mean=False)
Xtr_sc = scaler.fit_transform(X_train_final)
Xte_sc = scaler.transform(X_test_final)
lr_model = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
lr_model.fit(Xtr_sc, y_train)
p_lr = lr_model.predict_proba(Xte_sc)[:, 1]

# --- 训练 Random Forest (新增部分) ---
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=5, 
    random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
)
rf_model.fit(X_train_final, y_train)
p_rf = rf_model.predict_proba(X_test_final)[:, 1]

# 结果存储
store = {
    "Logistic Regression": {"y": y_test, "p": p_lr, "auc": roc_auc_score(y_test, p_lr)},
    "Random Forest":       {"y": y_test, "p": p_rf, "auc": roc_auc_score(y_test, p_rf)}
}

# RF Feature Importance
rf_imp = pd.DataFrame({"Feature": X_train_final.columns, "Importance": rf_model.feature_importances_}).sort_values("Importance", ascending=False)

# =========================================================
# 4) Dashboard 定义 (使用原始 Application Columns)
# =========================================================
app = dash.Dash(__name__)
server = app.server
add_basic_auth(server, AUTH_USER, AUTH_PASS)
app.title = "Risk Prediction Dashboard"

# 定义 Dashboard 输入字段 (这是为了让用户看原始字段名)
# 我们将从原始 app_df 中提取选项
raw_cat_cols = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", 
                "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]
raw_num_cols = ["AMT_INCOME_TOTAL", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "DAYS_BIRTH", "Account_Age_Months", "Active_Months"]

# 提取选项
options_map = {c: sorted([str(x) for x in app_df[c].dropna().unique()]) for c in raw_cat_cols if c in app_df.columns}

# 手动添加 Y/N 选项以防数据中缺失
options_map["FLAG_OWN_CAR"] = ["Y", "N"]
options_map["FLAG_OWN_REALTY"] = ["Y", "N"]

def generate_inputs():
    inputs = []
    
    # 1. Categorical Inputs
    inputs.append(html.H4("Personal Info (Categorical)", style={"marginTop": "20px"}))
    row = []
    for c in raw_cat_cols:
        if c not in options_map: continue
        comp = html.Div([
            html.Label(c, style={"fontWeight": "bold", "fontSize": "12px"}),
            dcc.Dropdown(id={'type': 'in', 'index': c}, options=[{'label': i, 'value': i} for i in options_map[c]], placeholder="Select...", style={"fontSize": "13px"})
        ], style={"width": "23%", "display": "inline-block", "marginRight": "2%", "marginBottom": "10px"})
        row.append(comp)
    inputs.append(html.Div(row))

    # 2. Numerical Inputs
    inputs.append(html.H4("Financial & Stats (Numeric)", style={"marginTop": "20px"}))
    row = []
    
    # 特殊处理：DAYS_BIRTH
    row.append(html.Div([
        html.Label("AGE (Years)", style={"fontWeight": "bold", "fontSize": "12px"}), # UI显示年龄
        dcc.Input(id={'type': 'in', 'index': 'DAYS_BIRTH'}, type="number", placeholder="e.g. 35", style={"width": "100%", "padding": "6px"})
    ], style={"width": "23%", "display": "inline-block", "marginRight": "2%"}))

    # 其他数值
    for c in ["AMT_INCOME_TOTAL", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]:
        row.append(html.Div([
            html.Label(c, style={"fontWeight": "bold", "fontSize": "12px"}),
            dcc.Input(id={'type': 'in', 'index': c}, type="number", placeholder="0", style={"width": "100%", "padding": "6px"})
        ], style={"width": "23%", "display": "inline-block", "marginRight": "2%"}))
    inputs.append(html.Div(row))

    # 3. Credit History (Required Inputs)
    inputs.append(html.H4("Credit Bureau Data (Simulated)", style={"marginTop": "20px"}))
    row = []
    for c in ["Account_Age_Months", "Active_Months"]:
        row.append(html.Div([
            html.Label(c, style={"fontWeight": "bold", "fontSize": "12px"}),
            dcc.Input(id={'type': 'in', 'index': c}, type="number", value=12, style={"width": "100%", "padding": "6px"})
        ], style={"width": "23%", "display": "inline-block", "marginRight": "2%"}))
    inputs.append(html.Div(row))
    
    return inputs

app.layout = html.Div([
    html.H2("Credit Risk Predictor: Raw Data Input"),
    
    html.Div([
        html.Label("Select Model:"),
        dcc.Dropdown(id="model_sel", options=[{"label": k, "value": k} for k in store.keys()], value="Random Forest", clearable=False, style={"width": "300px"})
    ], style={"marginBottom": "20px"}),

    dcc.Tabs([
        dcc.Tab(label="Model Performance", children=[
            html.Div([
                dcc.Graph(id="roc_graph", style={"width": "48%", "display": "inline-block"}),
                dcc.Graph(id="conf_matrix", style={"width": "48%", "display": "inline-block"}),
            ]),
            html.Div(id="imp_container", children=[
                html.H4("Random Forest Feature Importance"),
                dash_table.DataTable(
                    data=rf_imp.head(15).to_dict("records"),
                    columns=[{"name": i, "id": i} for i in rf_imp.columns],
                    style_cell={"textAlign": "left"},
                    page_size=10
                )
            ])
        ]),
        dcc.Tab(label="Prediction Simulator (Raw Data)", children=[
            html.Div(generate_inputs(), style={"padding": "20px", "backgroundColor": "#f9f9f9"}),
            html.Button("Predict Risk", id="btn_run", n_clicks=0, style={"marginTop": "20px", "fontSize": "18px", "padding": "10px 20px", "backgroundColor": "#007bff", "color": "white"}),
            html.Div(id="pred_result", style={"marginTop": "20px", "fontSize": "24px", "fontWeight": "bold"})
        ])
    ])
], style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "Arial"})

# =========================================================
# 5) Callbacks
# =========================================================

# A. 更新图表
@app.callback(
    [Output("roc_graph", "figure"), Output("conf_matrix", "figure"), Output("imp_container", "style")],
    Input("model_sel", "value")
)
def update_charts(model_name):
    d = store[model_name]
    fpr, tpr, _ = roc_curve(d["y"], d["p"])
    
    fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, fill='tozeroy'))
    fig_roc.update_layout(title=f"ROC Curve (AUC={d['auc']:.3f})", xaxis_title="FPR", yaxis_title="TPR")
    
    # 简单的混淆矩阵 (Threshold 0.5)
    cm = confusion_matrix(d["y"], (d["p"]>=0.5).astype(int))
    fig_cm = px.imshow(cm, text_auto=True, title=f"Confusion Matrix ({model_name})", labels=dict(x="Pred", y="True"))
    
    # 控制 Feature Importance 显示
    style = {"display": "block"} if model_name == "Random Forest" else {"display": "none"}
    
    return fig_roc, fig_cm, style

# B. 处理预测 (核心逻辑)
@app.callback(
    Output("pred_result", "children"),
    Input("btn_run", "n_clicks"),
    State({'type': 'in', 'index': ALL}, 'value'),
    State({'type': 'in', 'index': ALL}, 'id'),
    State("model_sel", "value"),
    prevent_initial_call=True
)
def run_prediction(n, values, ids, model_name):
    # 1. 收集原始输入
    raw_data = {item['index']: val for val, item in zip(values, ids)}
    df_raw = pd.DataFrame([raw_data])
    
    # 简单校验
    if df_raw.isnull().all().all():
        return "Please input data."

    # 处理特殊输入：用户输入的 Age (e.g., 30) 需要视为 DAYS_BIRTH
    # 在 engineer_features 中通过负数判断，这里为了兼容，我们把用户输入的正数转为负数？
    # 不，我在 engineer_features 里写了逻辑： "如果是正数，保持不变"。
    # 但是 DAYS_BIRTH 原始是负数。为了保持模型一致性，最好我们在 engineer_features 内部统一处理。
    # 修正逻辑：Dashboard 传入 Age=30，engineer_features 看到正数30，直接用作 Age_Clean=30 即可。
    # 因为 engineer_features 里写的是: if x < 0 abs(x)... else x. 
    # 所以只要输入的是正数年龄，逻辑是通的。
    
    try:
        # 2. 调用特征工程 (Raw -> Features)
        # 这一步会自动计算 Asset_Score, Adult_Income_Efficiency 等
        df_engineered = engineer_features(df_raw)
        
        # 3. 预处理 (Impute -> OHE -> Poly)
        # 注意：ColumnTransformer 会忽略 DataFrame 中多余的列（raw columns），只取 transform 需要的列
        Z_input = preprocess.transform(df_engineered)
        
        # 4. 选择特征 (Poly Filter)
        Z_input_sel = pd.concat([
            Z_input[non_poly_cols], 
            Z_input[poly_pure_cols]
        ], axis=1)
        
        # 5. 最终列对齐
        X_final = Z_input_sel[final_cols]
        
        # 6. 预测
        if model_name == "Logistic Regression":
            X_sc = scaler.transform(X_final)
            prob = lr_model.predict_proba(X_sc)[0, 1]
        else:
            prob = rf_model.predict_proba(X_final)[0, 1]
            
        color = "red" if prob > 0.5 else "green"
        return html.Div([
            f"Risk Probability: {prob:.2%}", 
            html.Br(),
            html.Span("High Risk" if prob > 0.5 else "Low Risk", style={"color": color})
        ])
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run_server(host="0.0.0.0", port=port, debug=True)

