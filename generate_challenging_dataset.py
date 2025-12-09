"""Generate a more challenging dataset to test realistic metrics"""
import pandas as pd
from sklearn.datasets import make_classification
import numpy as np

# Create a CHALLENGING classification dataset with:
# - More noise
# - Overlapping classes
# - Redundant features
X, y = make_classification(
    n_samples=1500,
    n_features=20,
    n_informative=8,      # Only 8 out of 20 features are useful
    n_redundant=5,        # 5 features are redundant
    n_repeated=2,         # 2 features are repeated
    n_classes=3,
    n_clusters_per_class=1,  # Less separation between classes
    class_sep=0.5,        # LOW separation (makes it harder)
    flip_y=0.15,          # 15% label noise (realistic)
    random_state=42
)

# Add some random noise features
noise_features = np.random.randn(1500, 5)
X = np.hstack([X, noise_features])

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(25)])
df['target'] = y

# Save to CSV
df.to_csv('challenging_dataset.csv', index=False)
print(f"Created challenging dataset with {len(df)} rows and {len(df.columns)} columns")
print(f"Target distribution:\n{df['target'].value_counts()}")
print("\nThis dataset has:")
print("- 15% label noise")
print("- Low class separation (0.5)")
print("- 5 noise features")
print("- Only 8 truly informative features out of 25")
print("\nExpected accuracy: 60-80% (realistic!)")
