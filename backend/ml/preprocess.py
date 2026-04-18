import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def remove_id_columns(df, target_column):
    """
    Automatically identify and remove non-informative identifier columns.
    
    Drops columns that:
    1. Are named "id" (case-insensitive)
    2. Have unique values for every row (nunique == len(df))
    
    Never drops the target_column.
    """
    cols_to_drop = []
    
    for col in df.columns:
        if col == target_column:
            continue
            
        # Check 1: Named "id"
        if col.lower() == 'id':
            cols_to_drop.append(col)
            continue
            
        # Check 2: All unique values
        if df[col].nunique() == len(df):
            cols_to_drop.append(col)
            
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Removed ID-like columns: {cols_to_drop}")
        
    return df

def preprocess_data(X_train, X_test):
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Fit on training data and transform both training and testing data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor