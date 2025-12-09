"""Generate a sample dataset for testing the ML dashboard"""
import pandas as pd
from sklearn.datasets import make_classification
import numpy as np

# Create a realistic classification dataset
X, y = make_classification(
    n_samples=800,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=2,
    random_state=42,
    flip_y=0.1  # Add some noise to make it realistic
)

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(12)])
df['target'] = y

# Save to CSV
df.to_csv('sample_dataset.csv', index=False)
print(f"Created sample dataset with {len(df)} rows and {len(df.columns)} columns")
print(f"Target distribution:\n{df['target'].value_counts()}")
