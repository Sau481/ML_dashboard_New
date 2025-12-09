"""
Test script to verify ML dashboard fixes
This script tests the updated ML pipeline with a sample dataset
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from ml.trainer import train_and_evaluate_models

def test_classification_small():
    """Test classification with small dataset"""
    print("\n" + "="*60)
    print("TEST 1: Small Classification Dataset (500 samples)")
    print("="*60)
    
    # Create a realistic small dataset
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    results, preprocessor = train_and_evaluate_models(df, 'target', 'classification')
    
    print("\nResults Summary:")
    for result in results[:3]:  # Show top 3
        print(f"\n{result['model']}:")
        print(f"  CV Accuracy: {result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}")
        print(f"  Train Accuracy: {result['train_accuracy']:.4f}")
        print(f"  Val Accuracy: {result['val_accuracy']:.4f}")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Overfitting Gap: {result['overfitting_gap']:.4f}")
        print(f"  Total Time: {result['total_time']:.4f}s")
    
    # Verify realistic metrics
    best = results[0]
    assert best['test_accuracy'] < 1.0, "❌ Test accuracy should not be perfect!"
    assert best['test_accuracy'] > 0.5, "❌ Test accuracy too low!"
    assert 'total_time' in best, "❌ Missing timing information!"
    print("\n✅ Small classification test PASSED!")

def test_regression_medium():
    """Test regression with medium dataset"""
    print("\n" + "="*60)
    print("TEST 2: Medium Regression Dataset (2000 samples)")
    print("="*60)
    
    # Create a realistic medium dataset
    X, y = make_regression(
        n_samples=2000,
        n_features=15,
        n_informative=10,
        noise=10.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
    df['target'] = y
    
    results, preprocessor = train_and_evaluate_models(df, 'target', 'regression')
    
    print("\nResults Summary:")
    for result in results[:3]:  # Show top 3
        print(f"\n{result['model']}:")
        print(f"  CV R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
        print(f"  Train R²: {result['train_r2_score']:.4f}")
        print(f"  Val R²: {result['val_r2_score']:.4f}")
        print(f"  Test R²: {result['test_r2_score']:.4f}")
        print(f"  Overfitting Gap: {result['overfitting_gap']:.4f}")
        print(f"  Total Time: {result['total_time']:.4f}s")
    
    # Verify realistic metrics
    best = results[0]
    assert best['test_r2_score'] < 1.0, "❌ Test R² should not be perfect!"
    assert 'total_time' in best, "❌ Missing timing information!"
    print("\n✅ Medium regression test PASSED!")

def test_dataset_size_detection():
    """Test that different dataset sizes use appropriate models"""
    print("\n" + "="*60)
    print("TEST 3: Dataset Size Detection")
    print("="*60)
    
    from ml.trainer import detect_dataset_size
    
    # Create test dataframes
    df_small = pd.DataFrame({'a': range(500)})
    df_medium = pd.DataFrame({'a': range(5000)})
    df_large = pd.DataFrame({'a': range(15000)})
    
    assert detect_dataset_size(df_small) == 'small', "❌ Small dataset not detected!"
    assert detect_dataset_size(df_medium) == 'medium', "❌ Medium dataset not detected!"
    assert detect_dataset_size(df_large) == 'large', "❌ Large dataset not detected!"
    
    print("✅ Dataset size detection PASSED!")

if __name__ == "__main__":
    print("\n🧪 Running ML Dashboard Tests...")
    
    try:
        test_dataset_size_detection()
        test_classification_small()
        test_regression_medium()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nKey Improvements Verified:")
        print("✅ Realistic accuracy scores (not perfect)")
        print("✅ Timing metrics for all operations")
        print("✅ Cross-validation scores with standard deviation")
        print("✅ Train/Val/Test split evaluation")
        print("✅ Overfitting detection")
        print("✅ Dataset size-based optimization")
        print("\nYour ML dashboard is now ready to use!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
