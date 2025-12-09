from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error, confusion_matrix,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model_name, model, X_train, y_train, X_val, y_val, X_test, y_test, problem_type, n_folds=5, dataset_size='medium'):
    """
    Comprehensive model evaluation with timing and cross-validation
    
    Args:
        model_name: Name of the model
        model: Model instance
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        X_test: Test features
        y_test: Test target
        problem_type: 'classification' or 'regression'
        n_folds: Number of cross-validation folds
        dataset_size: 'small', 'medium', or 'large'
    
    Returns:
        Dictionary with comprehensive evaluation metrics and timing
    """
    
    # Training time
    train_start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - train_start
    
    # Prediction time on validation set
    val_pred_start = time.time()
    y_val_pred = model.predict(X_val)
    val_prediction_time = time.time() - val_pred_start
    
    # Prediction time on test set
    test_pred_start = time.time()
    y_test_pred = model.predict(X_test)
    test_prediction_time = time.time() - test_pred_start
    
    # Cross-validation on training data (for more realistic performance estimate)
    cv_start = time.time()
    if problem_type == 'classification':
        cv_scoring = 'accuracy'
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        cv_scoring = 'r2'
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=cv_scoring, n_jobs=1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except Exception as e:
        cv_mean = 0.0
        cv_std = 0.0
        print(f"CV failed for {model_name}: {e}")
    
    cv_time = time.time() - cv_start
    
    # Evaluate on validation and test sets
    if problem_type == 'classification':
        result = evaluate_classification(
            model_name, model, 
            X_train, y_train,
            X_val, y_val, y_val_pred,
            X_test, y_test, y_test_pred,
            cv_mean, cv_std
        )
    else:
        result = evaluate_regression(
            model_name, model,
            X_train, y_train,
            X_val, y_val, y_val_pred,
            X_test, y_test, y_test_pred,
            cv_mean, cv_std
        )
    
    # Add timing information
    result['training_time'] = round(training_time, 4)
    result['val_prediction_time'] = round(val_prediction_time, 4)
    result['test_prediction_time'] = round(test_prediction_time, 4)
    result['cv_time'] = round(cv_time, 4)
    result['total_time'] = round(training_time + val_prediction_time + test_prediction_time + cv_time, 4)
    
    return result

def evaluate_classification(model_name, model, X_train, y_train, X_val, y_val, y_val_pred, X_test, y_test, y_test_pred, cv_mean, cv_std):
    """Evaluate classification model on validation and test sets"""
    
    # Training set metrics (to detect overfitting)
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Validation set metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    
    # Test set metrics (final evaluation)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    # Confusion Matrix
    test_confusion_matrix = confusion_matrix(y_test, y_test_pred).tolist()
    
    # ROC AUC and PR-AUC if available
    val_roc_auc = None
    test_roc_auc = None
    val_pr_auc = None
    test_pr_auc = None
    
    if hasattr(model, "predict_proba"):
        try:
            y_val_proba = model.predict_proba(X_val)
            y_test_proba = model.predict_proba(X_test)
            
            if y_val_proba.shape[1] == 2:
                # Binary classification
                val_roc_auc = roc_auc_score(y_val, y_val_proba[:, 1])
                test_roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
                val_pr_auc = average_precision_score(y_val, y_val_proba[:, 1])
                test_pr_auc = average_precision_score(y_test, y_test_proba[:, 1])
            else:
                # Multi-class classification
                val_roc_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr')
                test_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr')
                val_pr_auc = average_precision_score(y_val, y_val_proba, average='weighted')
                test_pr_auc = average_precision_score(y_test, y_test_proba, average='weighted')
        except Exception:
            pass
    
    return {
        'model': model_name,
        # Cross-validation scores
        'cv_accuracy_mean': round(cv_mean, 4),
        'cv_accuracy_std': round(cv_std, 4),
        # Training set (to detect overfitting)
        'train_accuracy': round(train_accuracy, 4),
        # Validation set
        'val_accuracy': round(val_accuracy, 4),
        'val_precision': round(val_precision, 4),
        'val_recall': round(val_recall, 4),
        'val_f1_score': round(val_f1, 4),
        'val_roc_auc': round(val_roc_auc, 4) if val_roc_auc else None,
        # Test set (final metrics)
        'test_accuracy': round(test_accuracy, 4),
        'test_precision': round(test_precision, 4),
        'test_recall': round(test_recall, 4),
        'test_f1_score': round(test_f1, 4),
        'test_roc_auc': round(test_roc_auc, 4) if test_roc_auc else None,
        'test_pr_auc': round(test_pr_auc, 4) if test_pr_auc else None,
        # Confusion Matrix
        'confusion_matrix': test_confusion_matrix,
        # Overfitting indicator
        'overfitting_gap': round(train_accuracy - test_accuracy, 4)
    }

def evaluate_regression(model_name, model, X_train, y_train, X_val, y_val, y_val_pred, X_test, y_test, y_test_pred, cv_mean, cv_std):
    """Evaluate regression model on validation and test sets"""
    
    # Training set metrics (to detect overfitting)
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Validation set metrics
    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    
    # Test set metrics (final evaluation)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    
    # Calculate Adjusted R² for test set
    n = len(y_test)  # number of samples
    p = X_test.shape[1]  # number of features
    test_adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1) if n > p + 1 else None
    
    # Calculate Adjusted R² for validation set
    n_val = len(y_val)
    val_adj_r2 = 1 - (1 - val_r2) * (n_val - 1) / (n_val - p - 1) if n_val > p + 1 else None
    
    return {
        'model': model_name,
        # Cross-validation scores
        'cv_r2_mean': round(cv_mean, 4),
        'cv_r2_std': round(cv_std, 4),
        # Training set (to detect overfitting)
        'train_r2_score': round(train_r2, 4),
        # Validation set
        'val_r2_score': round(val_r2, 4),
        'val_adj_r2_score': round(val_adj_r2, 4) if val_adj_r2 is not None else None,
        'val_mae': round(val_mae, 4),
        'val_mse': round(val_mse, 4),
        'val_rmse': round(val_rmse, 4),
        # Test set (final metrics)
        'test_r2_score': round(test_r2, 4),
        'test_adj_r2_score': round(test_adj_r2, 4) if test_adj_r2 is not None else None,
        'test_mae': round(test_mae, 4),
        'test_mse': round(test_mse, 4),
        'test_rmse': round(test_rmse, 4),
        # Overfitting indicator
        'overfitting_gap': round(train_r2 - test_r2, 4)
    }