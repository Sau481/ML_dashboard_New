from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

def with_scaling(model):
    """Wrap model with StandardScaler using Pipeline"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def get_models(problem_type, dataset_size='medium'):
    """
    Get models with configurations optimized for dataset size and production best practices

    Args:
        problem_type: 'classification' or 'regression'
        dataset_size: 'small' (<1000), 'medium' (1000-10000), or 'large' (>10000)

    Returns:
        Dictionary of model instances
    """

    if problem_type == 'classification':
        if dataset_size == 'small':
            return {
                'Logistic Regression': with_scaling(LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', class_weight='balanced', random_state=42)),
                'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=5, class_weight='balanced', random_state=42),
                'KNN': with_scaling(KNeighborsClassifier(n_neighbors=5)),
                'XGBoost': XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
            }
        elif dataset_size == 'medium':
            return {
                'Logistic Regression': with_scaling(LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)),
                'SVM': with_scaling(SVC(probability=True, kernel='rbf', C=1.0, class_weight='balanced', random_state=42)),
                'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
                'KNN': with_scaling(KNeighborsClassifier(n_neighbors=7)),
                'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
            }
        else:  # large
            return {
                'Logistic Regression': with_scaling(LogisticRegression(max_iter=500, C=1.0, solver='saga', class_weight='balanced', random_state=42)),
                'Decision Tree': DecisionTreeClassifier(max_depth=15, min_samples_split=20, class_weight='balanced', random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=1, class_weight='balanced', n_jobs=-1, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
                'KNN': with_scaling(KNeighborsClassifier(n_neighbors=10, n_jobs=-1)),
                'XGBoost': XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
            }

    else:  # regression
        if dataset_size == 'small':
            return {
                'Linear Regression': with_scaling(LinearRegression()),
                'Ridge': with_scaling(Ridge(alpha=1.0, random_state=42)),
                'Lasso': with_scaling(Lasso(alpha=1.0, max_iter=1000, random_state=42)),
                'Decision Tree': DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42),
                'KNN': with_scaling(KNeighborsRegressor(n_neighbors=5)),
                'XGBoost': XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
            }
        elif dataset_size == 'medium':
            return {
                'Linear Regression': with_scaling(LinearRegression()),
                'Ridge': with_scaling(Ridge(alpha=1.0, random_state=42)),
                'Lasso': with_scaling(Lasso(alpha=1.0, max_iter=1000, random_state=42)),
                'SVR': with_scaling(SVR(kernel='rbf', C=1.0)),
                'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, n_jobs=-1, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
                'KNN': with_scaling(KNeighborsRegressor(n_neighbors=7)),
                'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            }
        else:  # large
            return {
                'Linear Regression': with_scaling(LinearRegression()),
                'Ridge': with_scaling(Ridge(alpha=1.0, random_state=42)),
                'Lasso': with_scaling(Lasso(alpha=1.0, max_iter=500, random_state=42)),
                'Decision Tree': DecisionTreeRegressor(max_depth=15, min_samples_split=20, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=1, n_jobs=-1, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
                'KNN': with_scaling(KNeighborsRegressor(n_neighbors=10, n_jobs=-1)),
                'XGBoost': XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, n_jobs=-1)
            }