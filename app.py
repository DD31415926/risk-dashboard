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
APP_PATH  = "application_record.csv"
CRED_PATH = "credit_record.csv"

TEST_SIZE = 0.3
RANDOM_STATE = 42

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
# 1) 核心特征工程逻辑
# =========================================================
def engineer_features(df_in):
    """
    接收用户列出的原始字段, 转换为模型所需的工程特征。
    """
    df = df_in.copy()
    
    # 1. MONTHS_BALANCE -> Account Age
    if "MONTHS_BALANCE" in df.columns:
        df["Account_Age_Months"] = pd.to_numeric(df["MONTHS_BALANCE"], errors='coerce').abs()
        df["Active_Months"] = df["Account_Age_Months"] # 简化假设
    elif "Account_Age_Months" not in df.columns:
        df["Account_Age_Months"] = 0
        df["Active_Months"] = 0

    # 2. Asset Score
    if "FLAG_OWN_CAR" in df.columns and "FLAG_OWN_REALTY" in df.columns:
        mapper = {'Y': 1, 'N': 0, 'YES': 1, 'NO': 0, '1': 1, '0': 0}
        has_car = df["FLAG_OWN_CAR"].astype(str).str.upper().map(mapper).fillna(0)
        has_realty = df["FLAG_OWN_REALTY"].astype(str).str.upper().map(mapper).fillna(0)
        df["Asset_Score"] = has_car + has_realty
    
    # 3. Age
    if "DAYS_BIRTH" in df.columns:
        d_birth = pd.to_numeric(df["DAYS_BIRTH"], errors='coerce')
        # 负数转正，正数保持
        df["Age_Clean"] = d_birth.apply(lambda x: abs(x)/365.25 if x < 0 else x)
    
    # 4. Employment
    if "DAYS_EMPLOYED" in df.columns:
        d_emp = pd.to_numeric(df["DAYS_EMPLOYED"], errors='coerce')
        df["DAYS_EMPLOYED_CLEAN"] = d_emp.apply(lambda x: 0 if x > 200000 else abs(x)/365.25)

    # 5. Income Efficiency
    cols = ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "CNT_CHILDREN"]
    if set(cols).issubset(df.columns):
        inc = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors='coerce').fillna(0)
        fam = pd.to_numeric(df["CNT_FAM_MEMBERS"], errors='coerce').fillna(1)
        child = pd.to_numeric(df["CNT_CHILDREN"], errors='coerce').fillna(0)
        denom = (fam - child).clip(lower=1)
        df["Adult_Income_Efficiency"] = inc / denom
    
    return df

# =========================================================
# 2) 数据加载与预处理 (训练阶段)
# =========================================================
print("Loading and Merging Data...")
app_df = pd.read_csv(APP_PATH)
cred_df = pd.read_csv(CRED_PATH)

# --- Labeling ---
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

# --- Merge ---
merged_df = app_df.merge(labels, on="ID", how="inner")

# 获取历史长度用于训练的一致性
acct_stats = cred_df.groupby("ID")["MONTHS_BALANCE"].min().reset_index()
merged_df = merged_df.merge(acct_stats, on="ID", how="left")

# --- Feature Engineering ---
model_ready_df = engineer_features(merged_df)

DROP_COLS = ["ID", "label"]
y = model_ready_df["label"]
X = model_ready_df.drop(columns=DROP_COLS, errors="ignore")

# =========================================================
# 3) 训练流水线
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 排除不需要的列 (如果有)
EXCLUDE_NUM = ["ID", "label"]
num_all = [c for c in X.select_dtypes(include=[np.number]).columns if c not in EXCLUDE_NUM]
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 这里的 cat_cols 应该包含 OCCUPATION_TYPE
print(f"Categorical Columns found: {cat_cols}")

num_bin = [c for c in num_all if X[c].nunique() <= 2]
num_poly_base = [c for c in num_all if X[c].nunique() > 2]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")), # 这里处理 Occupation 的 NaN
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

print("Training Pipeline...")
Z_train = preprocess.fit_transform(X_train)
Z_test = preprocess.transform(X_test)

# Feature Selection
poly_cols_all = [c for c in Z_train.columns if any(base in c for base in num_poly_base)]
poly_pure_cols = [c for c in poly_cols_all if " " not in c]
non_poly_cols = [c for c in Z_train.columns if c not in poly_cols_all]

Z_train_sel = pd.concat([Z_train[non_poly_cols], Z_train[poly_pure_cols]], axis=1)
Z_test_sel = pd.concat([Z_test[non_poly_cols], Z_test[poly_pure_cols]], axis=1)

# T-test Filter
def ttest_pvalue(x, y):
    try: return stats.ttest_ind(x[y==0], x[y==1], equal_var=False, nan_policy="omit")[1]
    except: return np.nan

pvals = [(c, ttest_pvalue(Z_train_sel[c], y_train)) for c in poly_pure_cols]
sig_poly = [c for c, p in pvals if p < 0.05]
if not sig_poly: sig_poly = [c for c, p in sorted(pvals, key=lambda x: x[1] if pd.notna(x[1]) else 1)[:20]]

final_cols = non_poly_cols + sig_poly
X_train_final = Z_train_sel[final_cols]
X_test_final = Z_test_sel[final_cols]

# Models
scaler = StandardScaler(with_mean=False)
Xtr_sc = scaler.fit_transform(X_train_final)
Xte_sc = scaler.transform(X_test_final)

lr_model = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
lr_model.fit(Xtr_sc, y_train)
p_lr = lr_model.predict_proba(Xte_sc)[:, 1]

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=5, 
    random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
)
rf_model.fit(X_train_final, y_train)
p_rf = rf_model.predict_proba(X_test_final)[:, 1]

store = {
    "Logistic Regression": {"y": y_test.values, "p": p_lr, "auc": roc_auc_score(y_test, p_lr), "ap": average_precision_score(y_test, p_lr)},
    "Random Forest":       {"y": y_test.values, "p": p_rf, "auc": roc_auc_score(y_test, p_rf), "ap": average_precision_score(y_test, p_rf)}
}

rf_imp = pd.DataFrame({"Feature": X_train_final.columns, "Importance": rf_model.feature_importances_}).sort_values("Importance", ascending=False)

# =========================================================
# 4) Dashboard
# =========================================================
app = dash.Dash(__name__)
server = app.server
add_basic_auth(server, AUTH_USER, AUTH_PASS)
app.title = "Raw Data Predictor"

# 定义需要的原始字段 (包括 Occupation Type)
RAW_CAT_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", 
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", 
    "OCCUPATION_TYPE", # <--- 这里加回来了
    "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"
]

RAW_NUM_COLS = [
    "AMT_INCOME_TOTAL", "CNT_CHILDREN", "CNT_FAM_MEMBERS", 
    "DAYS_BIRTH", "DAYS_EMPLOYED", "MONTHS_BALANCE"
]

# 提取选项
options_map = {}
for c in RAW_CAT_COLS:
    if c in app_df.columns:
        opts = sorted([str(x) for x in app_df[c].dropna().unique()])
        options_map[c] = opts
    else:
        options_map[c] = ["0", "1"]

# 修正 Flags
for flag in ["FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
    if flag in options_map:
        curr = set(options_map[flag])
        if "0" not in curr and "1" not in curr:
             options_map[flag] = ["0", "1"]
        else:
             options_map[flag] = sorted(list(curr))

def generate_raw_inputs():
    inputs = []
    
    # 1. Categorical
    inputs.append(html.H4("Categorical Info", style={"marginTop": "10px", "borderBottom": "1px solid #eee"}))
    row = []
    for c in RAW_CAT_COLS:
        # 特别处理 Occupation 的 placeholder
        ph = "Select Occupation" if c == "OCCUPATION_TYPE" else f"Select {c}"
        
        comp = html.Div([
            html.Label(c, style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(
                id={'type': 'in', 'index': c}, 
                options=[{'label': i, 'value': i} for i in options_map.get(c, [])], 
                placeholder=ph,
                style={"fontSize": "12px"}
            )
        ], style={"width": "23%", "display": "inline-block", "marginRight": "2%", "marginBottom": "15px"})
        row.append(comp)
    inputs.append(html.Div(row))

    # 2. Numerical
    inputs.append(html.H4("Financial & Stats", style={"marginTop": "10px", "borderBottom": "1px solid #eee"}))
    row = []
    help_text = {
        "DAYS_BIRTH": "e.g. 33 (Years) or -12000 (Days)",
        "DAYS_EMPLOYED": "e.g. 5 (Years) or -1800 (Days)",
        "MONTHS_BALANCE": "e.g. -12 (History Length)"
    }
    
    for c in RAW_NUM_COLS:
        comp = html.Div([
            html.Label(c, style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id={'type': 'in', 'index': c}, 
                type="number", 
                placeholder=help_text.get(c, "0"),
                style={"width": "100%", "padding": "6px", "fontSize": "12px"}
            )
        ], style={"width": "23%", "display": "inline-block", "marginRight": "2%", "marginBottom": "15px"})
        row.append(comp)
    inputs.append(html.Div(row))
    
    return inputs

app.layout = html.Div([
    html.H2("Credit Risk Predictor: Raw Data Input"),
    
    html.Div([
        html.Div([
            html.Label("Select Model:"),
            dcc.Dropdown(id="model_sel", options=[{"label": k, "value": k} for k in store.keys()], value="Random Forest", clearable=False)
        ], style={"width": "30%", "display": "inline-block"}),
        
        html.Div([
            html.Label("Threshold:"),
            dcc.Slider(id="threshold", min=0, max=1, step=0.01, value=0.5, marks={0: "0", 0.5: "0.5", 1: "1"})
        ], style={"width": "60%", "display": "inline-block", "marginLeft": "5%", "verticalAlign": "top"})
    ], style={"padding": "15px", "backgroundColor": "#f8f9fa", "marginBottom": "20px"}),

    dcc.Tabs([
        dcc.Tab(label="Model Evaluation", children=[
            html.Div([
                html.Div([
                    dcc.Graph(id="conf_matrix"),
                    html.Div(id="metrics_text", style={"padding": "15px", "border": "1px solid #ddd", "marginTop": "10px", "backgroundColor": "white"})
                ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top"}),

                html.Div([
                    dcc.Graph(id="roc_curve"),
                    dcc.Graph(id="pr_curve")
                ], style={"width": "58%", "display": "inline-block", "marginLeft": "2%"})
            ], style={"padding": "20px"}),
            
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

        dcc.Tab(label="Prediction Simulator", children=[
            html.Div([
                html.P("Input raw client data. If Occupation is unknown, leave blank.", style={"color": "#666"}),
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
    
    cm = [[tn, fp], [fn, tp]]
    fig_cm = px.imshow(cm, text_auto=True, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                       title=f"Confusion Matrix ({model_name})", color_continuous_scale="Blues")
    fig_cm.update_layout(coloraxis_showscale=False)
    
    txt = [
        html.Div([html.Span("Accuracy: ", style={"fontWeight": "bold"}), f"{acc:.2%}"]),
        html.Div([html.Span("Precision: ", style={"fontWeight": "bold"}), f"{prec:.2%}"]),
        html.Div([html.Span("Recall: ", style={"fontWeight": "bold"}), f"{rec:.2%}"]),
        html.Div([html.Span("F1 Score: ", style={"fontWeight": "bold"}), f"{f1:.3f}"]),
        html.Br(),
        html.Div([html.Span("ROC AUC: ", style={"fontWeight": "bold"}), f"{d['auc']:.3f}"]),
        html.Div([html.Span("Avg Precision: ", style={"fontWeight": "bold"}), f"{d['ap']:.3f}"])
    ]
    
    fpr, tpr, _ = roc_curve(d["y"], d["p"])
    fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, name="ROC", fill='tozeroy'))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig_roc.update_layout(title="ROC Curve", height=300, margin=dict(t=30, b=20, l=40, r=20))
    
    p_arr, r_arr, _ = precision_recall_curve(d["y"], d["p"])
    fig_pr = go.Figure(go.Scatter(x=r_arr, y=p_arr, name="PR", fill='tozeroy', line=dict(color="orange")))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision",
                         height=300, margin=dict(t=30, b=20, l=40, r=20))
    
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
    raw_data = {item['index']: val for val, item in zip(values, ids)}
    df_raw = pd.DataFrame([raw_data])
    
    if df_raw.isnull().all().all():
        return "Please input data."
    
    try:
        # 补全缺失列 (例如用户没选 Occupation, 则为 NaN)
        for col in RAW_CAT_COLS + RAW_NUM_COLS:
            if col not in df_raw.columns:
                df_raw[col] = np.nan
        
        # 处理流程
        df_eng = engineer_features(df_raw)
        Z_in = preprocess.transform(df_eng) # Pipeline 处理 NaN
        
        Z_sel = pd.concat([Z_in[non_poly_cols], Z_in[poly_pure_cols]], axis=1)
        X_final = Z_sel[final_cols]
        
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



