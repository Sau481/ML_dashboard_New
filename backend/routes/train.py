from fastapi import APIRouter, HTTPException, Request, Form
import os
import uuid
import joblib
from typing import Optional
from backend.utils import read_csv_from_file, detect_problem_type, templates
from backend.ml.trainer import train_and_evaluate_models
from backend.schemas.request_response import TrainRequest, TrainResponse, ModelResult

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

@router.post("/train")
async def train(
    request: Request,
    file_path: Optional[str] = Form(None),
    target: Optional[str] = Form(None),
    problem_type_form: Optional[str] = Form(None, alias="problem_type"),
    train_req: Optional[TrainRequest] = None
):
    # Handle both Form and JSON input
    if file_path and target:
        f_path = file_path
        t_col = target
        p_type = problem_type_form
    elif train_req:
        f_path = train_req.file_path
        t_col = train_req.target
        p_type = train_req.problem_type
    else:
        # Try to parse JSON body if not already done
        try:
            body = await request.json()
            f_path = body.get("file_path")
            t_col = body.get("target")
            p_type = body.get("problem_type")
        except:
            raise HTTPException(status_code=400, detail="Missing training parameters")

    try:
        df = read_csv_from_file(f_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    if t_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{t_col}' not found in dataset")
    
    # Use manual selection if provided, otherwise use auto-detection
    if not p_type:
        detected_type, _ = detect_problem_type(df[t_col])
        p_type = detected_type

    try:
        results, preprocessor = train_and_evaluate_models(df, t_col, p_type)
        
        # Save the preprocessor
        preprocessor_path = os.path.join(RESULTS_DIR, f"{uuid.uuid4().hex}.joblib")
        joblib.dump(preprocessor, preprocessor_path)

        # Check if the request wants HTML or JSON
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return templates.TemplateResponse(
                request=request,
                name="results.html",
                context={
                    "results": results,
                    "problem_type": p_type
                }
            )

        # Convert results to ModelResult pydantic models
        model_results = [ModelResult(**res) for res in results]
        return TrainResponse(
            results=model_results,
            problem_type=p_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during training: {str(e)}")
