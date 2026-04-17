from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error, confusion_matrix,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

def get_feature_importance(model, feature_names):
    """Extract and normalize feature importance from various model types"""
    # If it's a pipeline, get the model from the last step
    if isinstance(model, Pipeline):
        inner_model = model.named_steps['model']
    else:
        inner_model = model
    
    importance = None
    try:
        if hasattr(inner_model, 'feature_importances_'):
            importance = inner_model.feature_importances_
        elif hasattr(inner_model, 'coef_'):
            importance = np.abs(inner_model.coef_)
            if len(importance.shape) > 1:
                importance = np.mean(importance, axis=0)
        
        if importance is not None:
            # Handle potential dimension mismatch if any
            if len(importance) != len(feature_names):
                return None
            
            # Normalize to sum to 1
            sum_importance = np.sum(importance)
            if sum_importance > 0:
                importance = importance / sum_importance
                
            # Create a dictionary and sort by importance
            feat_imp = dict(zip(feature_names, [round(float(i), 4) for i in importance]))
            return dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        pass
    return None

def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, problem_type, feature_names, n_folds=5):
    """
    Comprehensive model evaluation with timing and cross-validation
    
    Args:
        model_name: Name of the model
        model: Model instance (or Pipeline)
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        problem_type: 'classification' or 'regression'
        feature_names: List of feature names
        n_folds: Number of cross-validation folds
    
    Returns:
        Dictionary with comprehensive evaluation metrics and timing
    """
    
    # Training time
    train_start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - train_start
    
    # Prediction time on test set
    test_pred_start = time.time()
    y_test_pred = model.predict(X_test)
    test_prediction_time = time.time() - test_pred_start
    
    # Aliasing for UI compatibility
    val_prediction_time = test_prediction_time 
    
    # Cross-validation on training data
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
    
    # Feature importance
    feature_importance = get_feature_importance(model, feature_names)
    
    # Evaluate on test set
    if problem_type == 'classification':
        result = evaluate_classification(
            model_name, model, 
            X_train, y_train,
            X_test, y_test, y_test_pred,
            cv_mean, cv_std
        )
    else:
        result = evaluate_regression(
            model_name, model,
            X_train, y_train,
            X_test, y_test, y_test_pred,
            cv_mean, cv_std
        )
    
    # Add timing and feature importance
    result['training_time'] = round(training_time, 4)
    result['test_prediction_time'] = round(test_prediction_time, 4)
    result['val_prediction_time'] = round(val_prediction_time, 4)
    result['cv_time'] = round(cv_time, 4)
    result['total_time'] = round(training_time + test_prediction_time + cv_time, 4)
    result['feature_importance'] = feature_importance
    
    return result

def evaluate_classification(model_name, model, X_train, y_train, X_test, y_test, y_test_pred, cv_mean, cv_std):
    """Evaluate classification model on training and test sets"""
    
    # Training set metrics (to detect overfitting)
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Test set metrics (final evaluation)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    # Confusion Matrix
    test_confusion_matrix = confusion_matrix(y_test, y_test_pred).tolist()
    
    # ROC AUC and PR-AUC if available
    test_roc_auc = None
    test_pr_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_test_proba = model.predict_proba(X_test)
            if y_test_proba.shape[1] == 2:
                test_roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
                test_pr_auc = average_precision_score(y_test, y_test_proba[:, 1])
            else:
                test_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr')
                test_pr_auc = average_precision_score(y_test, y_test_proba, average='weighted')
        except Exception:
            pass
    
    return {
        'model': model_name,
        'cv_score_mean': round(cv_mean, 4),
        'cv_score_std': round(cv_std, 4),
        # Legacy support for frontend
        'cv_accuracy_mean': round(cv_mean, 4),
        'cv_accuracy_std': round(cv_std, 4),
        
        'train_accuracy': round(train_accuracy, 4),
        'test_accuracy': round(test_accuracy, 4),
        'val_accuracy': round(test_accuracy, 4), # Use test for val in UI
        
        'test_precision': round(test_precision, 4),
        'test_recall': round(test_recall, 4),
        'test_f1_score': round(test_f1, 4),
        'test_roc_auc': round(test_roc_auc, 4) if test_roc_auc else None,
        'test_pr_auc': round(test_pr_auc, 4) if test_pr_auc else None,
        'confusion_matrix': test_confusion_matrix,
        'overfitting_gap': round(train_accuracy - test_accuracy, 4)
    }

def evaluate_regression(model_name, model, X_train, y_train, X_test, y_test, y_test_pred, cv_mean, cv_std):
    """Evaluate regression model on training and test sets"""
    
    # Training set metrics (to detect overfitting)
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test set metrics (final evaluation)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    
    return {
        'model': model_name,
        'cv_score_mean': round(cv_mean, 4),
        'cv_score_std': round(cv_std, 4),
        # Legacy support
        'cv_r2_mean': round(cv_mean, 4),
        'cv_r2_std': round(cv_std, 4),
        
        'train_r2_score': round(train_r2, 4),
        'test_r2_score': round(test_r2, 4),
        'val_r2_score': round(test_r2, 4), # Use test for val in UI
        
        'test_mae': round(test_mae, 4),
        'test_mse': round(test_mse, 4),
        'test_rmse': round(test_rmse, 4),
        'overfitting_gap': round(train_r2 - test_r2, 4)
    }
