import pandas as pd
import os
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "frontend", "templates")

templates = Jinja2Templates(directory=TEMPLATE_DIR)
templates.env.globals['get_flashed_messages'] = lambda: []

def read_csv_from_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def basic_overview(df):
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().astype(int).to_dict(),
        "sample": df.head(5).to_dict(orient="records")
    }

def detect_problem_type(y):
    """
    Smart detection of problem type (classification vs regression)
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
