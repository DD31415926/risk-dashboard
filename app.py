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
# 0) 配置（生产部署时建议用环境变量覆盖）
# =========================================================
CSV_PATH = os.environ.get("CSV_PATH", "model_data2(4).csv")
TARGET_COL = os.environ.get("TARGET_COL", "label")
ID_COL = os.environ.get("ID_COL", "ID")

TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.30"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))

# 可选：Basic Auth（强烈建议上线后开启）
# Render/Railway 的环境变量里设置：
#   DASH_AUTH_USER=xxx
#   DASH_AUTH_PASS=yyy
AUTH_USER = os.environ.get("DASH_AUTH_USER")
AUTH_PASS = os.environ.get("DASH_AUTH_PASS")


# =========================================================
# 1) 可选：Basic Auth（不额外依赖 dash-auth）
#    - 如果没设置 DASH_AUTH_USER/PASS，则不启用鉴权
# =========================================================
def add_basic_auth(server, username, password):
    if not username or not password:
        return

    from flask import request, Response

    @server.before_request
    def _basic_auth():
        auth = request.authorization
        ok = (auth is not None and auth.username == username and auth.password == password)
        if ok:
            return None
        return Response(
            "Authentication required", 401,
            {"WWW-Authenticate": 'Basic realm="Login Required"'}
        )


# =========================================================
# 2) 读数据 + holdout 切分（第二段模式）
# =========================================================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}. Make sure it is included in deployment.")

df = pd.read_csv(CSV_PATH)

y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL], errors="ignore")
if ID_COL in X.columns:
    X = X.drop(columns=[ID_COL], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# =========================================================
# 3) 列分组
# =========================================================
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_all = X.select_dtypes(include=[np.number]).columns.tolist()
num_bin = [c for c in num_all if X[c].nunique(dropna=True) <= 2]
num_poly_base = [c for c in num_all if X[c].nunique(dropna=True) > 2]

cat_options = {col: sorted(list(X[col].dropna().unique())) for col in cat_cols}

# =========================================================
# 4) preprocess（train fit）
# =========================================================
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


# train fit + transform
Z_train = preprocess.fit_transform(X_train)
Z_test = preprocess.transform(X_test)

# 第二段同款：识别 poly 列、去交互项、拼接 Z_train2/Z_test2
poly_cols = [c for c in Z_train.columns if any(base in c for base in num_poly_base)]
poly_no_inter_cols = [c for c in poly_cols if (" " not in c)]
non_poly_cols = [c for c in Z_train.columns if c not in poly_cols]

Z_train2 = pd.concat([Z_train[non_poly_cols], Z_train[poly_no_inter_cols]], axis=1)
Z_test2 = pd.concat([Z_test[non_poly_cols], Z_test[poly_no_inter_cols]], axis=1)

# t-test 只在 train 上做
pvals = [(c, ttest_pvalue(Z_train2[c], y_train)) for c in poly_no_inter_cols]
pvals_df = pd.DataFrame(pvals, columns=["feature", "p_value"]).sort_values("p_value")

sig_poly = pvals_df[(pvals_df["p_value"].notna()) & (pvals_df["p_value"] < 0.05)]["feature"].tolist()
if len(sig_poly) == 0:
    sig_poly = pvals_df.dropna().head(20)["feature"].tolist()

final_cols = non_poly_cols + sig_poly

X_train_final = Z_train2[final_cols]
X_test_final = Z_test2[final_cols]

# =========================================================
# 5) 训练两模型（第二段同款）
# =========================================================
# LR
scaler = StandardScaler(with_mean=False)
Xtr_sc = scaler.fit_transform(X_train_final)
Xte_sc = scaler.transform(X_test_final)

final_logit = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")
final_logit.fit(Xtr_sc, y_train)
p_lr = final_logit.predict_proba(Xte_sc)[:, 1]
auc_lr = roc_auc_score(y_test, p_lr)
ap_lr = average_precision_score(y_test, p_lr)

# RF
final_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)
final_rf.fit(X_train_final, y_train)
p_rf = final_rf.predict_proba(X_test_final)[:, 1]
auc_rf = roc_auc_score(y_test, p_rf)
ap_rf = average_precision_score(y_test, p_rf)

store = {
    "Logistic Regression": {"y_true": y_test.values, "y_proba": p_lr, "auc": auc_lr, "ap": ap_lr},
    "Random Forest": {"y_true": y_test.values, "y_proba": p_rf, "auc": auc_rf, "ap": ap_rf},
}

rf_importance = (
    pd.DataFrame({"feature": X_train_final.columns, "importance": final_rf.feature_importances_})
    .sort_values("importance", ascending=False)
)
stats_df = pd.DataFrame([{
    "Feature": f,
    "Type": "Used in Model",
    "RF Importance": f"{imp:.4f}"
} for f, imp in zip(rf_importance["feature"], rf_importance["importance"])])


# =========================================================
# 6) Dash UI
# =========================================================
def build_input_fields(columns, cat_cols, cat_opts):
    fields, row = [], []
    for col in columns:
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
            html.Label(col, style={"fontWeight": "bold", "fontSize": "12px"}),
            input_comp
        ], style={"width": "30%", "display": "inline-block", "marginRight": "3%", "marginBottom": "10px"}))

        if len(row) == 3:
            fields.append(html.Div(row))
            row = []
    if row:
        fields.append(html.Div(row))
    return fields


app = dash.Dash(__name__)
server = app.server  # 给 gunicorn 用
add_basic_auth(server, AUTH_USER, AUTH_PASS)

app.title = "Risk Model Dashboard (Holdout: LR & RF)"
input_fields_layout = build_input_fields(X.columns, cat_cols, cat_options)

app.layout = html.Div([
    html.H2("Credit Risk Dashboard: LR vs Random Forest (Holdout Test)",
            style={"borderBottom": "2px solid #333"}),

    html.Div([
        html.Div([
            html.Label("Select Model"),
            dcc.Dropdown(
                id="model_name",
                options=[{"label": k, "value": k} for k in store.keys()],
                value="Random Forest",
                clearable=False
            )
        ], style={"width": "25%", "display": "inline-block"}),

        html.Div([
            html.Label("Threshold"),
            dcc.Slider(id="threshold", min=0, max=1, step=0.01, value=0.5,
                       marks={0: "0", 0.5: "0.5", 1: "1"})
        ], style={"width": "70%", "display": "inline-block", "marginLeft": "5%"})
    ], style={"padding": "15px", "backgroundColor": "#f9f9f9", "marginBottom": "20px"}),

    dcc.Tabs([
        dcc.Tab(label="Evaluation (Holdout Test)", children=[
            html.Div([
                html.Div([
                    dcc.Graph(id="conf_matrix"),
                    html.Div(id="metrics_text",
                             style={"padding": "10px", "border": "1px solid #ddd", "marginTop": "10px"})
                ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top"}),

                html.Div([
                    dcc.Graph(id="roc_curve"),
                    dcc.Graph(id="pr_curve")
                ], style={"width": "58%", "display": "inline-block", "marginLeft": "2%"})
            ], style={"padding": "20px"}),

            html.H4("Feature Importance (Random Forest; trained on TRAIN split)"),
            dash_table.DataTable(
                data=stats_df.head(20).to_dict("records"),
                columns=[{"name": i, "id": i} for i in stats_df.columns],
                page_size=10,
                style_cell={"textAlign": "left"}
            )
        ]),

        dcc.Tab(label="Prediction Simulator", children=[
            html.Div([
                html.H4("Input Data for Prediction"),
                html.P("Preprocessing + poly(no interactions) + selected features are applied exactly as training."),
                html.Div(id="input_container", children=input_fields_layout),
                html.Button("Predict Risk", id="btn_predict", n_clicks=0,
                            style={"fontSize": "16px", "marginTop": "15px", "padding": "10px 20px",
                                   "backgroundColor": "#007bff", "color": "white"}),
                html.Hr(),
                html.Div(id="prediction_output",
                         style={"fontSize": "20px", "fontWeight": "bold", "padding": "10px"})
            ], style={"padding": "20px", "maxWidth": "1000px", "margin": "0 auto"})
        ])
    ])
], style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "Arial"})


def compute_metrics(y_true, y_proba, thr):
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / len(y_true)
    return tn, fp, fn, tp, prec, rec, f1, acc


@app.callback(
    Output("conf_matrix", "figure"),
    Output("metrics_text", "children"),
    Output("roc_curve", "figure"),
    Output("pr_curve", "figure"),
    Input("model_name", "value"),
    Input("threshold", "value")
)
def update_eval(model_name, thr):
    d = store[model_name]
    thr = float(thr)

    tn, fp, fn, tp, prec, rec, f1, acc = compute_metrics(d["y_true"], d["y_proba"], thr)

    cm = [[tn, fp], [fn, tp]]
    fig_cm = px.imshow(cm, text_auto=True, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                       title=f"Confusion Matrix (Holdout Test) — {model_name}")

    txt = [
        html.Div(f"Accuracy: {acc:.3f}"),
        html.Div(f"Precision: {prec:.3f}"),
        html.Div(f"Recall: {rec:.3f}"),
        html.Div(f"F1: {f1:.3f}"),
        html.Div(f"AUC: {d['auc']:.3f}"),
        html.Div(f"AP: {d['ap']:.3f}")
    ]

    fpr, tpr, _ = roc_curve(d["y_true"], d["y_proba"])
    fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, name="Model"))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig_roc.update_layout(title=f"ROC (Holdout Test) — {model_name}", height=300, margin=dict(t=40, b=20))

    p_arr, r_arr, _ = precision_recall_curve(d["y_true"], d["y_proba"])
    fig_pr = go.Figure(go.Scatter(x=r_arr, y=p_arr, name="Model"))
    fig_pr.update_layout(title=f"PR Curve (Holdout Test) — {model_name}", height=300, margin=dict(t=40, b=20))

    return fig_cm, txt, fig_roc, fig_pr


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
    row_dict = {comp_id["index"]: val for val, comp_id in zip(values, ids)}
    input_df = pd.DataFrame([row_dict])

    try:
        # preprocess transform
        Z_input = preprocess.transform(input_df)
        Z_input2 = pd.concat([Z_input[non_poly_cols], Z_input[poly_no_inter_cols]], axis=1)
        X_input_final = Z_input2[final_cols]

        if model_name == "Logistic Regression":
            X_ready = scaler.transform(X_input_final)
            proba = float(final_logit.predict_proba(X_ready)[0, 1])
        else:
            proba = float(final_rf.predict_proba(X_input_final)[0, 1])

        thr = float(thr)
        lbl = "High Risk" if proba >= thr else "Low Risk"
        color = "red" if proba >= thr else "green"

        return html.Div([
            html.Div(f"Model: {model_name}"),
            html.Div(f"Probability: {proba:.2%}"),
            html.Div(f"Prediction: {lbl}", style={"color": color, "fontWeight": "bold"})
        ])
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={"color": "red"})


# 本地运行（部署时用 gunicorn，不走这里）
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run_server(host="0.0.0.0", port=port, debug=True)
