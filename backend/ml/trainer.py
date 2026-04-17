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

def train_and_evaluate_models(df, target_column, problem_type):
    """
    Train and evaluate multiple models with production-level best practices
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        problem_type: 'classification' or 'regression'
    
    Returns:
        results: List of model evaluation results
        preprocessor: Fitted preprocessing pipeline
    """
    start_time = time.time()
    
    # Detect dataset size for optimization
    dataset_size = detect_dataset_size(df)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target labels if classification and non-numeric
    label_encoder = None
    if problem_type == 'classification':
        if y.dtype == 'object' or not pd.api.types.is_integer_dtype(y):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            print(f"Encoded target labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Strict train/test split (80/20)
    if problem_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        n_folds = 5 if dataset_size != 'small' else 3
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        n_folds = 5 if dataset_size != 'small' else 3
    
    # Preprocess data (Fit on TRAIN only to avoid leakage)
    preprocess_start = time.time()
    X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test)
    preprocess_time = time.time() - preprocess_start
    
    # Get feature names for importance analysis
    feature_names = get_feature_names(preprocessor, X_train_processed)
    
    # Get models based on dataset size
    models = get_models(problem_type, dataset_size)
    
    # Evaluate models in parallel
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
        result['preprocessing_time'] = round(preprocess_time, 4)
        result['dataset_size'] = dataset_size
        result['train_samples'] = len(X_train)
        result['test_samples'] = len(X_test)
        result['val_samples'] = 0  # Placeholder for UI
    
    # Best model selection logic based on CV score
    results.sort(key=lambda x: x.get('cv_score_mean', 0), reverse=True)
    
    if results:
        for i, r in enumerate(results):
            r['is_best'] = (i == 0)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline Upgrade Complete")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Best Model: {results[0]['model']} (CV Score: {results[0]['cv_score_mean']})")
    print(f"{'='*60}\n")
    
    return results, preprocessor
