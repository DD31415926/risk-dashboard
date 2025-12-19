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
# 请确保这两个文件在你的运行目录下，或者修改为绝对路径
APP_PATH = "/home/yaolushen/fall_242A/242A_datasets/application_record.csv"
CRED_PATH = "/home/yaolushen/fall_242A/242A_datasets/credit_record.csv"

# 用于生产环境的配置
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.30"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))

# Basic Auth 配置
AUTH_USER = os.environ.get("DASH_AUTH_USER")
AUTH_PASS = os.environ.get("DASH_AUTH_PASS")


# =========================================================
# 1) 工具函数：Basic Auth & T-test
# =========================================================
def add_basic_auth(server, username, password):
    if not username or not password:
        return
    from flask import request, Response
    @server.before_request
    def _basic_auth():
        auth = request.authorization
        ok = (auth is not None and auth.username == username and auth.password == password)
        if ok: return None
        return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="Login Required"'})


def ttest_pvalue(x, y_series):
    g0 = x[y_series == 0]
    g1 = x[y_series == 1]
    if pd.Series(x).nunique(dropna=True) <= 1:
        return np.nan
    try:
        _, p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
        return float(p)
    except Exception:
        return np.nan


# =========================================================
# 2) 数据处理流水线 (Data Loading & Engineering)
# =========================================================
print("Loading and Processing Data... This may take a moment.")

# --- Step A: 读取原始数据 ---
if not os.path.exists(APP_PATH) or not os.path.exists(CRED_PATH):
    raise FileNotFoundError(f"Data files not found. Ensure {APP_PATH} and {CRED_PATH} exist.")

app_df = pd.read_csv(APP_PATH)
cred_df = pd.read_csv(CRED_PATH)

# --- Step B: 定义标签 (Labeling) ---
WINDOW = 12
FULL_LEN = WINDOW + 1
BAD_SET = {"1", "2", "3", "4", "5"}

w = cred_df[
    (cred_df["MONTHS_BALANCE"] >= -WINDOW) &
    (cred_df["MONTHS_BALANCE"] <= 0)
    ].copy()

months_in_window = w.groupby("ID")["MONTHS_BALANCE"].nunique()
ever_bad = w.groupby("ID")["STATUS"].apply(lambda s: s.isin(BAD_SET).any())
ever_c = w.groupby("ID")["STATUS"].apply(lambda s: (s == "C").any())

lab = pd.DataFrame({
    "ID": months_in_window.index,
    "months_in_window": months_in_window.values,
    "ever_bad": ever_bad.reindex(months_in_window.index).values,
    "ever_c": ever_c.reindex(months_in_window.index).values,
})

lab["label"] = np.nan
full = lab["months_in_window"] >= FULL_LEN
lab.loc[full & (lab["ever_bad"]), "label"] = 1
lab.loc[full & (~lab["ever_bad"]) & (lab["ever_c"]), "label"] = 0

labels = lab[["ID", "label"]].dropna().copy()
labels["label"] = labels["label"].astype(int)

# 合并标签
final_df = app_df.merge(labels, on="ID", how="inner").copy()

# --- Step C: 手动特征工程 (Manual Feature Engineering) ---
# 1. Account Age (Credit History)
account_age = cred_df.groupby("ID")["MONTHS_BALANCE"].agg(["min", "count"]).reset_index()
account_age["Account_Age_Months"] = account_age["min"].abs()
account_age["Active_Months"] = account_age["count"]
final_df = final_df.merge(account_age[["ID", "Account_Age_Months", "Active_Months"]], on="ID", how="left")

# 2. Employment
if "DAYS_EMPLOYED" in final_df.columns:
    final_df["DAYS_EMPLOYED_CLEAN"] = final_df["DAYS_EMPLOYED"].apply(
        lambda x: 0 if pd.notna(x) and x > 0 else (abs(x) / 365.25 if pd.notna(x) else np.nan)
    )

# 3. Age
if "DAYS_BIRTH" in final_df.columns:
    final_df["Age_Clean"] = final_df["DAYS_BIRTH"].abs() / 365.25

# 4. Asset Score
if "FLAG_OWN_CAR" in final_df.columns and "FLAG_OWN_REALTY" in final_df.columns:
    final_df["Asset_Score"] = (final_df["FLAG_OWN_CAR"].astype(str) == "Y").astype(int) + \
                              (final_df["FLAG_OWN_REALTY"].astype(str) == "Y").astype(int)

# 5. Income Efficiency
if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "CNT_CHILDREN"}.issubset(final_df.columns):
    denom = (final_df["CNT_FAM_MEMBERS"] - final_df["CNT_CHILDREN"]).clip(lower=1)
    final_df["Adult_Income_Efficiency"] = final_df["AMT_INCOME_TOTAL"] / denom

# --- Step D: 清理不需要的原始列以避免共线性 ---
# 我们保留 engineered features，丢弃原始的 DAYS_BIRTH 等，以免模型混淆
cols_to_drop = ["ID", "label", "DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_MOBIL",
                "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"]
# 注意：保留 AMT_INCOME_TOTAL 等基础数值列，因为它们本身也是特征

y = final_df["label"].astype(int)
X = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns], errors="ignore")

# =========================================================
# 3) 数据切分与多项式流水线
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 列定义
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_all = X.select_dtypes(include=[np.number]).columns.tolist()
num_bin = [c for c in num_all if X[c].nunique(dropna=True) <= 2]
num_poly_base = [c for c in num_all if X[c].nunique(dropna=True) > 2]

# 为 Dashboard 准备下拉选项
cat_options = {col: sorted([str(x) for x in X[col].dropna().unique()]) for col in cat_cols}

# Pipeline 定义
poly = PolynomialFeatures(degree=2, include_bias=False)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
        ("num_bin", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_bin),
        ("num_poly", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", poly),
        ]), num_poly_base),
    ],
    remainder="drop",
    verbose_feature_names_out=False
).set_output(transform="pandas")

# Fit & Transform
print("Running Pipeline and Poly features...")
Z_train = preprocess.fit_transform(X_train)
Z_test = preprocess.transform(X_test)

# 去除交互项 (Interaction Terms)，只保留 x 和 x^2
poly_cols_all = [c for c in Z_train.columns if any(base in c for base in num_poly_base)]
poly_no_inter_cols = [c for c in poly_cols_all if (" " not in c)]  # 简单规则：无空格即为纯项(如 x0^2)
non_poly_cols = [c for c in Z_train.columns if c not in poly_cols_all]

Z_train2 = pd.concat([Z_train[non_poly_cols], Z_train[poly_no_inter_cols]], axis=1)
Z_test2 = pd.concat([Z_test[non_poly_cols], Z_test[poly_no_inter_cols]], axis=1)

# T-test 特征筛选
print("Performing T-test selection...")
pvals = [(c, ttest_pvalue(Z_train2[c], y_train)) for c in poly_no_inter_cols]
pvals_df = pd.DataFrame(pvals, columns=["feature", "p_value"]).sort_values("p_value")
sig_poly = pvals_df[(pvals_df["p_value"].notna()) & (pvals_df["p_value"] < 0.05)]["feature"].tolist()

# 兜底策略
if len(sig_poly) == 0:
    sig_poly = pvals_df.dropna().head(20)["feature"].tolist()

final_cols = non_poly_cols + sig_poly

X_train_final = Z_train2[final_cols]
X_test_final = Z_test2[final_cols]

print(f"Final feature count: {X_train_final.shape[1]}")

# =========================================================
# 4) 模型训练 (LR & RF)
# =========================================================
# --- Logistic Regression ---
scaler = StandardScaler(with_mean=False)
Xtr_sc = scaler.fit_transform(X_train_final)
Xte_sc = scaler.transform(X_test_final)

print("Training Logistic Regression...")
final_logit = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")
final_logit.fit(Xtr_sc, y_train)

p_lr = final_logit.predict_proba(Xte_sc)[:, 1]
auc_lr = roc_auc_score(y_test, p_lr)
ap_lr = average_precision_score(y_test, p_lr)

# --- Random Forest (补充部分) ---
print("Training Random Forest...")
final_rf = RandomForestClassifier(
    n_estimators=300,  # 适度树数量
    max_depth=10,  # 限制深度防止过拟合
    min_samples_leaf=5,  # 增加叶子节点样本数要求
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)
# RF 不需要 Scaling，直接用筛选后的特征
final_rf.fit(X_train_final, y_train)

p_rf = final_rf.predict_proba(X_test_final)[:, 1]
auc_rf = roc_auc_score(y_test, p_rf)
ap_rf = average_precision_score(y_test, p_rf)

# 存储结果供 Dashboard 调用
store = {
    "Logistic Regression": {"y_true": y_test.values, "y_proba": p_lr, "auc": auc_lr, "ap": ap_lr},
    "Random Forest": {"y_true": y_test.values, "y_proba": p_rf, "auc": auc_rf, "ap": ap_rf},
}

# 提取 Feature Importance
rf_importance = (
    pd.DataFrame({"feature": X_train_final.columns, "importance": final_rf.feature_importances_})
    .sort_values("importance", ascending=False)
)
# 格式化表格数据
stats_df = pd.DataFrame([{
    "Feature": f,
    "Importance": f"{imp:.4f}"
} for f, imp in zip(rf_importance["feature"], rf_importance["importance"])])


# =========================================================
# 5) Dash UI 构建
# =========================================================
def build_input_fields(columns, cat_cols, cat_opts):
    fields, row = [], []
    # 简单的排序，把数值和类别分开
    sorted_cols = sorted(columns)

    for col in sorted_cols:
        if col in cat_cols:
            input_comp = dcc.Dropdown(
                id={'type': 'pred-input', 'index': col},
                options=[{'label': str(v), 'value': v} for v in cat_opts.get(col, [])],
                placeholder=f"Select {col}",
                style={"fontSize": "13px"}
            )
        else:
            input_comp = dcc.Input(
                id={'type': 'pred-input', 'index': col},
                type="number",
                placeholder=col,
                debounce=True,
                style={"width": "100%", "padding": "6px", "boxSizing": "border-box"}
            )

        row.append(html.Div([
            html.Label(col,
                       style={"fontWeight": "bold", "fontSize": "12px", "display": "block", "marginBottom": "5px"}),
            input_comp
        ], style={"width": "30%", "display": "inline-block", "marginRight": "3%", "marginBottom": "15px",
                  "verticalAlign": "top"}))

        if len(row) == 3:
            fields.append(html.Div(row))
            row = []
    if row:
        fields.append(html.Div(row))
    return fields


app = dash.Dash(__name__)
server = app.server
add_basic_auth(server, AUTH_USER, AUTH_PASS)

app.title = "Credit Risk Dashboard"

# 使用 X.columns 构建输入框 (包含 engineered features 如 Age_Clean)
# 这是折中方案：用户直接输入 "Clean Age" 比输入 "Birth Date" 更符合模型逻辑，且不需要复杂的交互转换
input_fields_layout = build_input_fields(X.columns, cat_cols, cat_options)

app.layout = html.Div([
    html.H2("Credit Risk Dashboard: End-to-End Pipeline",
            style={"borderBottom": "2px solid #333", "paddingBottom": "10px"}),

    # 控制栏
    html.Div([
        html.Div([
            html.Label("Select Model", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="model_name",
                options=[{"label": k, "value": k} for k in store.keys()],
                value="Random Forest",
                clearable=False
            )
        ], style={"width": "25%", "display": "inline-block"}),

        html.Div([
            html.Label("Decision Threshold", style={"fontWeight": "bold"}),
            dcc.Slider(id="threshold", min=0, max=1, step=0.01, value=0.5,
                       marks={0: "0", 0.5: "0.5", 1: "1"})
        ], style={"width": "60%", "display": "inline-block", "marginLeft": "5%", "verticalAlign": "top"})
    ], style={"padding": "20px", "backgroundColor": "#f1f3f4", "borderRadius": "5px", "marginBottom": "25px"}),

    dcc.Tabs([
        # Tab 1: 模型评估
        dcc.Tab(label="Model Evaluation", children=[
            html.Div([
                # 左侧：混淆矩阵 + 指标
                html.Div([
                    dcc.Graph(id="conf_matrix"),
                    html.Div(id="metrics_text",
                             style={"padding": "15px", "border": "1px solid #ddd", "borderRadius": "5px",
                                    "backgroundColor": "white", "marginTop": "10px"})
                ], style={"width": "38%", "display": "inline-block", "verticalAlign": "top"}),

                # 右侧：ROC & PR 曲线
                html.Div([
                    dcc.Graph(id="roc_curve"),
                    dcc.Graph(id="pr_curve")
                ], style={"width": "58%", "display": "inline-block", "marginLeft": "2%"})
            ], style={"padding": "20px"}),

            # Feature Importance (仅 RF 展示)
            html.Div(id="feat_imp_wrapper", children=[
                html.H4("Feature Importance (Top 20 - Random Forest Only)", style={"marginTop": "30px"}),
                dash_table.DataTable(
                    id="feat_imp_table",
                    data=stats_df.head(20).to_dict("records"),
                    columns=[{"name": i, "id": i} for i in stats_df.columns],
                    page_size=10,
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_cell={"textAlign": "left", "padding": "10px"}
                )
            ], style={"padding": "20px"})
        ]),

        # Tab 2: 模拟预测
        dcc.Tab(label="Prediction Simulator", children=[
            html.Div([
                html.H4("Input Features"),
                html.P(
                    "Enter values for the features below. Note: 'Age_Clean' is age in years. 'Account_Age_Months' is history length.",
                    style={"color": "#666", "fontSize": "14px"}),

                html.Div(id="input_container", children=input_fields_layout, style={"marginTop": "20px"}),

                html.Button("Predict Risk Score", id="btn_predict", n_clicks=0,
                            style={"fontSize": "16px", "marginTop": "20px", "padding": "12px 24px",
                                   "backgroundColor": "#28a745", "color": "white", "border": "none",
                                   "borderRadius": "4px", "cursor": "pointer"}),
                html.Hr(),
                html.Div(id="prediction_output",
                         style={"fontSize": "22px", "fontWeight": "bold", "padding": "15px", "minHeight": "60px"})
            ], style={"padding": "30px", "maxWidth": "1000px", "margin": "0 auto"})
        ])
    ])
], style={"maxWidth": "1280px", "margin": "0 auto", "fontFamily": "Segoe UI, Arial, sans-serif"})


# =========================================================
# 6) Callbacks
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
     Output("feat_imp_wrapper", "style")],  # 控制表格显示/隐藏
    [Input("model_name", "value"),
     Input("threshold", "value")]
)
def update_eval(model_name, thr):
    d = store[model_name]
    thr = float(thr)

    tn, fp, fn, tp, prec, rec, f1, acc = compute_metrics(d["y_true"], d["y_proba"], thr)

    # 1. Confusion Matrix
    cm = [[tn, fp], [fn, tp]]
    fig_cm = px.imshow(cm, text_auto=True, x=["Pred 0 (Safe)", "Pred 1 (Risk)"], y=["True 0", "True 1"],
                       color_continuous_scale="Blues",
                       title=f"Confusion Matrix ({model_name})")
    fig_cm.update_layout(coloraxis_showscale=False)

    # 2. Metrics Text
    txt = [
        html.Div([html.Span("Accuracy: ", style={"fontWeight": "bold"}), f"{acc:.2%}"]),
        html.Div([html.Span("Precision: ", style={"fontWeight": "bold"}), f"{prec:.2%}"]),
        html.Div([html.Span("Recall: ", style={"fontWeight": "bold"}), f"{rec:.2%}"]),
        html.Div([html.Span("F1 Score: ", style={"fontWeight": "bold"}), f"{f1:.3f}"]),
        html.Br(),
        html.Div([html.Span("ROC AUC: ", style={"fontWeight": "bold"}), f"{d['auc']:.3f}"]),
        html.Div([html.Span("Avg Precision: ", style={"fontWeight": "bold"}), f"{d['ap']:.3f}"])
    ]

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(d["y_true"], d["y_proba"])
    fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, name="Model", line=dict(width=3, color="#007bff")))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          height=320, margin=dict(t=40, b=20, l=40, r=20))

    # 4. PR Curve
    p_arr, r_arr, _ = precision_recall_curve(d["y_true"], d["y_proba"])
    fig_pr = go.Figure(go.Scatter(x=r_arr, y=p_arr, name="Model", line=dict(width=3, color="#28a745")))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision",
                         height=320, margin=dict(t=40, b=20, l=40, r=20))

    # 5. Hide Importance Table if LR
    table_style = {"display": "block", "padding": "20px"}
    if model_name == "Logistic Regression":
        table_style = {"display": "none"}

    return fig_cm, txt, fig_roc, fig_pr, table_style


@app.callback(
    Output("prediction_output", "children"),
    Input("btn_predict", "n_clicks"),
    State({"type": "pred-input", "index": ALL}, "value"),
    State({"type": "pred-input", "index": ALL}, "id"),
    State("model_name", "value"),
    State("threshold", "value"),
    prevent_initial_call=True
)
def predict_simulator(n_clicks, values, ids, model_name, thr):
    if not values or all(v is None for v in values):
        return html.Div("Please enter feature values.", style={"color": "gray"})

    # 组装输入 DataFrame
    row_dict = {comp_id["index"]: val for val, comp_id in zip(values, ids)}
    input_df = pd.DataFrame([row_dict])

    # 填补空值防止报错
    input_df.fillna(0, inplace=True)

    try:
        # 1. 预处理 (Pipeline: Impute -> OneHot -> Poly)
        Z_input = preprocess.transform(input_df)

        # 2. 特征对齐 (Poly filter)
        Z_input2 = pd.concat([Z_input[non_poly_cols], Z_input[poly_no_inter_cols]], axis=1)

        # 3. 最终特征选择
        X_input_final = Z_input2[final_cols]

        # 4. 预测
        if model_name == "Logistic Regression":
            X_ready = scaler.transform(X_input_final)
            proba = float(final_logit.predict_proba(X_ready)[0, 1])
        else:
            proba = float(final_rf.predict_proba(X_input_final)[0, 1])

        thr = float(thr)
        is_risk = proba >= thr
        lbl = "HIGH RISK (Denied)" if is_risk else "LOW RISK (Approved)"
        color = "#dc3545" if is_risk else "#28a745"

        return html.Div([
            html.Div(f"Model: {model_name} | Threshold: {thr}"),
            html.Div(f"Risk Probability: {proba:.2%}", style={"fontSize": "24px", "marginTop": "10px"}),
            html.Div(lbl, style={"color": color, "fontWeight": "bold", "fontSize": "30px", "marginTop": "5px"})
        ])
    except Exception as e:
        return html.Div(f"Prediction Error: {str(e)}", style={"color": "red"})


# =========================================================
# 7) 启动入口
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run_server(host="0.0.0.0", port=port, debug=True)
