import os
import uuid
import joblib
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash
)
import pandas as pd
from ml.trainer import train_and_evaluate_models

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
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().astype(int).to_dict(),
        "sample": df.head(5).to_dict(orient="records")
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

    # Determine problem type
    if df[target].dtype == 'object' or pd.api.types.is_categorical_dtype(df[target]):
        problem_type = "classification"
    else:
        problem_type = "regression"

    try:
        results, preprocessor = train_and_evaluate_models(df, target, problem_type)
        
        # Save the preprocessor
        preprocessor_path = os.path.join(RESULTS_DIR, f"{uuid.uuid4().hex}.joblib")
        joblib.dump(preprocessor, preprocessor_path)

    except Exception as e:
        flash(f"An error occurred during training: {e}")
        return redirect(url_for("home"))

    return render_template("results.html",
                           results=results,
                           problem_type=problem_type)

# ========================================================

if __name__ == "__main__":
    app.run(debug=True)
