import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .preprocess import preprocess_data
from .models import get_models
from .evaluator import evaluate_model
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

def detect_dataset_size(df):
    """Categorize dataset size for optimization"""
    n_rows = len(df)
    if n_rows < 1000:
        return 'small'
    elif n_rows < 10000:
        return 'medium'
    else:
        return 'large'

def get_feature_names(preprocessor, X):
    """Helper to get feature names after preprocessing"""
    try:
        if hasattr(preprocessor, 'get_feature_names_out'):
            return preprocessor.get_feature_names_out().tolist()
        return [f"Feature {i}" for i in range(X.shape[1])]
    except Exception:
        # Fallback if get_feature_names_out fails
        return [f"Feature {i}" for i in range(X.shape[1])]

def prepare_data(df, target_column, problem_type):
    """
    Standardized data preparation pipeline to ensure consistency.
    Includes data cleaning: removing ID-like columns.
    """
    print(f"\n--- DEBUG: Data Preparation ---")
    print(f"Original dataset shape: {df.shape}")
    print(f"Target column: {target_column}")
    print(f"Problem type: {problem_type}")

    # Data Cleaning: Remove ID-like columns
    initial_cols = set(df.columns)
    
    # 1. Remove obvious 'id' columns (case insensitive)
    id_cols = [col for col in df.columns if col.lower() in ['id', 'uuid', 'guid', 'index'] and col != target_column]
    df = df.drop(columns=id_cols)
    
    # 2. Remove columns where all values are unique (likely IDs or high-cardinality noise)
    high_cardinality_cols = []
    for col in df.columns:
        if col != target_column:
            if df[col].nunique() == len(df):
                high_cardinality_cols.append(col)
    
    df = df.drop(columns=high_cardinality_cols)
    
    removed_cols = list(initial_cols - set(df.columns))
    print(f"Removed columns: {removed_cols}")
    print(f"Cleaned dataset shape: {df.shape}")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target labels if classification and non-numeric
    if problem_type == 'classification':
        if y.dtype == 'object' or not pd.api.types.is_integer_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"Encoded target labels.")

    # Detect dataset size
    dataset_size = detect_dataset_size(df)

    # Strict train/test split (80/20)
    stratify = y if problem_type == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    
    # Preprocess data (Fit on TRAIN only to avoid leakage)
    X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test)
    
    print(f"Processed X_train sample (first 2 rows):\n{X_train_processed[:2]}")
    print(f"--- DEBUG END ---\n")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, dataset_size

def train_and_evaluate_models(df, target_column, problem_type):
    """
    Train and evaluate multiple models with production-level best practices
    """
    start_time = time.time()
    
    # Reuse standardized preparation
    X_train_processed, X_test_processed, y_train, y_test, preprocessor, dataset_size = \
        prepare_data(df, target_column, problem_type)
    
    preprocess_time = 0 # Preprocessing is now inside prepare_data, but for simplicity in results:
    
    # Get feature names
    feature_names = get_feature_names(preprocessor, X_train_processed)
    
    # Get models
    models = get_models(problem_type, dataset_size)
    
    # Evaluate models in parallel
    n_folds = 5 if dataset_size != 'small' else 3
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_model)(
            name, 
            model, 
            X_train_processed, 
            y_train, 
            X_test_processed, 
            y_test, 
            problem_type,
            feature_names,
            n_folds
        )
        for name, model in models.items()
    )
    
    # Post-process results
    for result in results:
        result['dataset_size'] = dataset_size
        result['train_samples'] = len(y_train)
        result['test_samples'] = len(y_test)
        result['val_samples'] = 0
        result['preprocessing_time'] = 0 # Metric support
    
    results.sort(key=lambda x: x.get('cv_score_mean', 0), reverse=True)
    if results:
        for i, r in enumerate(results):
            r['is_best'] = (i == 0)

    return results, preprocessor
