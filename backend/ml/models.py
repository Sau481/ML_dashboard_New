from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def get_models(problem_type, dataset_size='medium'):
    """
    Get models with configurations optimized for dataset size
    
    Args:
        problem_type: 'classification' or 'regression'
        dataset_size: 'small' (<1000), 'medium' (1000-10000), or 'large' (>10000)
    
    Returns:
        Dictionary of model instances
    """
    
    if problem_type == 'classification':
        if dataset_size == 'small':
            # Simpler models to prevent overfitting on small datasets
            return {
                'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
                'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5),
                'Random Forest': RandomForestClassifier(n_estimators=20, max_depth=5, min_samples_split=10, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5)
            }
        elif dataset_size == 'medium':
            # Balanced complexity for medium datasets
            return {
                'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
                'SVM': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
                'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=7)
            }
        else:  # large
            # Optimized for speed on large datasets
            return {
                'Logistic Regression': LogisticRegression(max_iter=500, C=1.0, solver='saga', random_state=42),
                'Decision Tree': DecisionTreeClassifier(max_depth=15, min_samples_split=20, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
            }
    
    else:  # regression
        if dataset_size == 'small':
            # Simpler models to prevent overfitting on small datasets
            return {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=1.0, max_iter=1000),
                'Decision Tree': DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=20, max_depth=5, min_samples_split=10, random_state=42),
                'KNN': KNeighborsRegressor(n_neighbors=5)
            }
        elif dataset_size == 'medium':
            # Balanced complexity for medium datasets
            return {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=1.0, max_iter=1000),
                'SVR': SVR(kernel='rbf', C=1.0),
                'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42),
                'KNN': KNeighborsRegressor(n_neighbors=7)
            }
        else:  # large
            # Optimized for speed on large datasets
            return {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=1.0, max_iter=500),
                'Decision Tree': DecisionTreeRegressor(max_depth=15, min_samples_split=20, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42),
                'KNN': KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
            }