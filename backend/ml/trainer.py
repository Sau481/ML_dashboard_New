import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
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

def train_and_evaluate_models(df, target_column, problem_type):
    """
    Train and evaluate multiple models with proper validation and timing
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        problem_type: 'classification' or 'regression'
    
    Returns:
        results: List of model evaluation results with timing
        preprocessor: Fitted preprocessing pipeline
    """
    start_time = time.time()
    
    # Detect dataset size for optimization
    dataset_size = detect_dataset_size(df)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Use stratified split for classification to maintain class distribution
    if problem_type == 'classification':
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
    
    # Preprocess data
    preprocess_start = time.time()
    X_train_processed, X_val_processed, preprocessor = preprocess_data(X_train, X_val)
    X_test_processed = preprocessor.transform(X_test)
    preprocess_time = time.time() - preprocess_start
    
    # Get models based on dataset size
    models = get_models(problem_type, dataset_size)
    
    # Determine number of CV folds based on dataset size
    if dataset_size == 'small':
        n_folds = min(3, len(X_train) // 10)  # At least 10 samples per fold
    elif dataset_size == 'medium':
        n_folds = 5
    else:
        n_folds = 3  # Fewer folds for large datasets to save time
    
    n_folds = max(2, n_folds)  # At least 2 folds
    
    # Evaluate models in parallel
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_model)(
            name, 
            model, 
            X_train_processed, 
            y_train, 
            X_val_processed, 
            y_val,
            X_test_processed, 
            y_test, 
            problem_type,
            n_folds,
            dataset_size
        )
        for name, model in models.items()
    )
    
    # Add preprocessing time to results
    for result in results:
        result['preprocessing_time'] = round(preprocess_time, 4)
        result['dataset_size'] = dataset_size
        result['train_samples'] = len(X_train)
        result['val_samples'] = len(X_val)
        result['test_samples'] = len(X_test)
    
    # Sort results by the primary metric
    if problem_type == 'classification':
        results.sort(key=lambda x: x.get('test_accuracy', 0), reverse=True)
    else:
        results.sort(key=lambda x: x.get('test_r2_score', 0), reverse=True)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Dataset size category: {dataset_size}")
    print(f"Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"{'='*60}\n")
    
    return results, preprocessor
