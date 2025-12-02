# backend/app.py
import os
import uuid
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_squared_error,
    mean_absolute_error, r2_score
)

# ---------------------------------------------------------
# FIXED PATHS FOR YOUR PROJECT STRUCTURE
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "frontend", "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "frontend", "static")

app = Flask(__name__,
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)

app.secret_key = "super-secret-key"

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================================================
# ---------------- HELPER FUNCTIONS ----------------------
# ========================================================

def read_csv_from_file(filepath):
    return pd.read_csv(filepath)

def basic_overview(df):
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),   # 🔥 ADDED BACK
        "missing_values": df.isnull().sum().astype(int).to_dict(),  # 🔥 matches template
        "sample": df.head(5).to_dict(orient="records")
    }

def detect_column_types(df, target):
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric = [c for c in numeric if c != target]
    categorical = [c for c in categorical if c != target]

    return numeric, categorical

def preprocess(df, target):
    df = df.copy()

    numeric_cols, cat_cols = detect_column_types(df, target)

    # Missing values
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    if cat_cols:
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode target
    target_is_cat = df[target].dtype == object
    label_encoder = None
    if target_is_cat:
        label_encoder = LabelEncoder()
        df[target] = label_encoder.fit_transform(df[target].astype(str))

    # One-hot encode
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Scale
    scaler = None
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    meta = {"target_is_categorical": target_is_cat}
    return X_train, X_test, y_train, y_test, meta, label_encoder, scaler

def get_models(problem_type):
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "SVR": SVR(),
            "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }

def evaluate_classification(y_true, y_pred, y_proba):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    except:
        metrics["roc_auc"] = None

    return metrics

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

# ========================================================
# -------------------- ROUTES ----------------------------
# ========================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if not file:
        flash("Upload a CSV file")
        return redirect(url_for("home"))

    file_id = uuid.uuid4().hex
    path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    file.save(path)

    df = read_csv_from_file(path)
    overview = basic_overview(df)

    return render_template("select_target.html",
                           file_path=path,
                           overview=overview)

@app.route("/train", methods=["POST"])
def train():
    file_path = request.form.get("file_path")
    target = request.form.get("target")

    df = read_csv_from_file(file_path)
    X_train, X_test, y_train, y_test, meta, le, scaler = preprocess(df, target)

    problem_type = "classification" if meta["target_is_categorical"] else "regression"

    models = get_models(problem_type)
    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if problem_type == "classification":
                y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                metrics = evaluate_classification(y_test, y_pred, y_proba)
            else:
                metrics = evaluate_regression(y_test, y_pred)

            results[name] = metrics

        except Exception as e:
            results[name] = {"error": str(e)}

    return render_template("results.html",
                           results=results,
                           problem_type=problem_type)

# ========================================================

if __name__ == "__main__":
    app.run(debug=True)
