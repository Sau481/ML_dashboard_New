"""
Diagnostic script to understand why you're getting high accuracy
This will test the actual pipeline and show you what's happening
"""

import pandas as pd
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from ml.trainer import train_and_evaluate_models

# Test with the challenging dataset
print("="*70)
print("TESTING WITH CHALLENGING DATASET")
print("="*70)

df = pd.read_csv('challenging_dataset.csv')
print(f"\nDataset loaded: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}\n")

results, preprocessor = train_and_evaluate_models(df, 'target', 'classification')

print("\n" + "="*70)
print("RESULTS ANALYSIS")
print("="*70)

for i, result in enumerate(results[:5], 1):
    print(f"\n{i}. {result['model']}")
    print(f"   CV Accuracy: {result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}")
    print(f"   Train Accuracy: {result['train_accuracy']:.4f}")
    print(f"   Val Accuracy: {result['val_accuracy']:.4f}")
    print(f"   Test Accuracy: {result['test_accuracy']:.4f}")
    print(f"   Overfitting Gap: {result['overfitting_gap']:.4f}")
    
    if result['test_accuracy'] > 0.95:
        print(f"   ⚠️  WARNING: Test accuracy is suspiciously high!")
    elif result['test_accuracy'] < 0.70:
        print(f"   ✅ GOOD: Realistic accuracy for challenging dataset")
    else:
        print(f"   ✓  OK: Moderate accuracy")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

best = results[0]
if best['test_accuracy'] > 0.95:
    print("❌ PROBLEM: Still getting unrealistically high accuracy!")
    print("\nPossible causes:")
    print("1. Dataset is too easy (even with noise)")
    print("2. Random state makes it reproducible")
    print("3. Models are too powerful for this dataset size")
elif best['test_accuracy'] < 0.70:
    print("✅ SUCCESS: Getting realistic accuracy scores!")
    print(f"\nBest model: {best['model']}")
    print(f"Test accuracy: {best['test_accuracy']:.2%}")
    print("The fixes are working correctly!")
else:
    print("⚠️  MODERATE: Accuracy is in reasonable range")
    print(f"Test accuracy: {best['test_accuracy']:.2%}")
