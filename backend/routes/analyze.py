from fastapi import APIRouter, HTTPException
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from backend.utils import read_csv_from_file, detect_problem_type
from backend.ml.models import get_models
from backend.ml.trainer import prepare_data
from backend.schemas.request_response import AnalyzeRequest, AnalyzeResponse
from backend.visualization import (
    generate_classification_plots, 
    generate_regression_plots, 
    generate_learning_curve
)
import numpy as np

router = APIRouter()

@router.post("/analyze-model", response_model=AnalyzeResponse)
async def analyze_model(request: AnalyzeRequest):
    try:
        df = read_csv_from_file(request.file_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {str(e)}")
        
    if request.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{request.target}' not found")
        
    # 1. Detect or use provided problem type
    if request.problem_type:
        problem_type = request.problem_type
    else:
        # We use the target column to detect problem type consistently
        problem_type, _ = detect_problem_type(df[request.target])
    
    # 2. Reuse standardized preparation logic
    X_train_processed, X_test_processed, y_train, y_test, preprocessor, dataset_size = \
        prepare_data(df, request.target, problem_type)
    
    # 3. Get models
    models = get_models(problem_type, dataset_size)
    
    # 4. Select ONLY requested model
    if request.model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not available. Choose from: {list(models.keys())}")
        
    model = models[request.model_name]
    
    # 5. Train model
    model.fit(X_train_processed, y_train)
    
    # 6. Predict
    y_pred = model.predict(X_test_processed)
    
    # Evaluation
    metrics = {}
    graphs = {}
    
    if problem_type == 'classification':
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        }
        
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_processed)
                if y_proba.shape[1] == 2:
                    metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)
                else:
                    metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba, multi_class='ovr'), 4)
            except:
                pass
                
        graphs = generate_classification_plots(model, X_test_processed, y_test, y_pred)
    else:
        metrics = {
            "r2_score": round(r2_score(y_test, y_pred), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        }
        graphs = generate_regression_plots(y_test, y_pred)
        
    # Learning Curve
    graphs['learning_curve'] = generate_learning_curve(model, X_train_processed, y_train, problem_type)
    
    return AnalyzeResponse(
        model=request.model_name,
        problem_type=problem_type,
        metrics=metrics,
        graphs=graphs
    )
