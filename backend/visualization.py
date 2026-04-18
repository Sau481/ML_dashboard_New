import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import learning_curve

def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_classification_plots(model, X_test, y_test, y_pred):
    graphs = {}
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    graphs['confusion_matrix'] = plot_to_base64()
    
    # ROC and PR Curves if model supports predict_proba
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            
            # ROC Curve (binary or macro-average for multi-class)
            plt.figure(figsize=(8, 6))
            if y_proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            else:
                # Simple OVR approach for visualization
                for i in range(y_proba.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            graphs['roc_curve'] = plot_to_base64()
            
            # Precision-Recall Curve
            plt.figure(figsize=(8, 6))
            if y_proba.shape[1] == 2:
                precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
                plt.plot(recall, precision)
            else:
                for i in range(y_proba.shape[1]):
                    precision, recall, _ = precision_recall_curve(y_test == i, y_proba[:, i])
                    plt.plot(recall, precision, label=f'Class {i}')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            graphs['precision_recall_curve'] = plot_to_base64()
        except:
            pass
            
    return graphs

def generate_regression_plots(y_test, y_pred):
    graphs = {}
    
    # Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    graphs['actual_vs_predicted'] = plot_to_base64()
    
    # Residual Plot
    plt.figure(figsize=(8, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    graphs['residual_plot'] = plot_to_base64()
    
    return graphs

def generate_learning_curve(model, X, y, problem_type):
    plt.figure(figsize=(8, 6))
    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring=scoring
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel(scoring.title())
    plt.title("Learning Curve")
    plt.legend(loc="best")
    
    return plot_to_base64()
