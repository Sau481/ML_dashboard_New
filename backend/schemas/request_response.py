from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class DatasetOverview(BaseModel):
    shape: List[int]
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    sample: List[Dict[str, Any]]

class ColumnMetadata(BaseModel):
    detected_type: str
    reason: str

class UploadResponse(BaseModel):
    file_path: str
    overview: DatasetOverview
    column_metadata: Dict[str, ColumnMetadata]

class ModelResult(BaseModel):
    model: str
    cv_score_mean: float
    cv_score_std: float
    # Classification metrics
    cv_accuracy_mean: Optional[float] = None
    cv_accuracy_std: Optional[float] = None
    train_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1_score: Optional[float] = None
    test_roc_auc: Optional[float] = None
    test_pr_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    # Regression metrics
    cv_r2_mean: Optional[float] = None
    cv_r2_std: Optional[float] = None
    train_r2_score: Optional[float] = None
    test_r2_score: Optional[float] = None
    val_r2_score: Optional[float] = None
    test_mae: Optional[float] = None
    test_mse: Optional[float] = None
    test_rmse: Optional[float] = None
    # Common
    overfitting_gap: float
    training_time: float
    test_prediction_time: float
    val_prediction_time: float
    cv_time: float
    preprocessing_time: float
    total_time: float
    dataset_size: str
    train_samples: int
    test_samples: int
    val_samples: int
    is_best: bool
    feature_importance: Optional[Dict[str, float]] = None
    
    # Visualization Data
    roc_curve: Optional[Dict[str, Any]] = None
    pr_curve: Optional[Dict[str, Any]] = None
    residual_plot: Optional[Dict[str, Any]] = None
    actual_vs_predicted: Optional[Dict[str, Any]] = None
    learning_curve: Optional[Dict[str, Any]] = None

class TrainResponse(BaseModel):
    results: List[ModelResult]
    problem_type: str

class TrainRequest(BaseModel):
    file_path: str
    target: str
    problem_type: Optional[str] = None

class AnalyzeRequest(BaseModel):
    file_path: str
    target: str
    model_name: str

class AnalyzeResponse(BaseModel):
    model: str
    problem_type: str
    metrics: Dict[str, Any]
    graphs: Dict[str, str]
