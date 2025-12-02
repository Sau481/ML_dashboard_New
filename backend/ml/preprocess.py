import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def preprocess_data(df):

    # Auto-select last column as target
    target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode target if categorical
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    # Scale numerical
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, target
