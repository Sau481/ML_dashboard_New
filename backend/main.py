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

def detect_problem_type(y):
    """
    Smart detection of problem type (classification vs regression)
    
    Args:
        y: Series or array-like target values
    
    Returns:
        tuple: (detected_type, reason)
    """
    # Ensure y is a pandas Series
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    unique_count = y.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(y)
    
    # Rule 1: If it's object or categorical type, it's classification
    if not is_numeric:
        return 'classification', f"Target column is non-numeric (type: {y.dtype})"
    
    # Rule 2: If numeric with <= 20 unique values, treat as classification
    if unique_count <= 20:
        return 'classification', f"Target is numeric but has only {unique_count} unique values (suggests categories)"
    
    # Rule 3: Otherwise, it's regression
    return 'regression', f"Target is numeric and has many ({unique_count}) unique values"

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
    
    # Add column metadata for auto-detection
    column_metadata = {}
    for col in df.columns:
        detected_type, reason = detect_problem_type(df[col])
        column_metadata[col] = {
            'detected_type': detected_type,
            'reason': reason
        }

    return render_template("select_target.html",
                           file_path=path,
                           overview=overview,
                           column_metadata=column_metadata)

@app.route("/train", methods=["POST"])
def train():
    file_path = request.form.get("file_path")
    target = request.form.get("target")
    manual_problem_type = request.form.get("problem_type")  # Get manual selection

    df = read_csv_from_file(file_path)

    # Use manual selection if provided, otherwise use auto-detection
    if manual_problem_type:
        problem_type = manual_problem_type
    else:
        # Smart auto-detection
        detected_type, _ = detect_problem_type(df[target])
        problem_type = detected_type

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
