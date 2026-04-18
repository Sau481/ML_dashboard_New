from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import uuid
import os
from backend.utils import read_csv_from_file, basic_overview, detect_problem_type, templates
from backend.schemas.request_response import UploadResponse, ColumnMetadata

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    file_id = uuid.uuid4().hex
    path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    
    try:
        with open(path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        df = read_csv_from_file(path)
        overview = basic_overview(df)
        
        # Add column metadata for auto-detection
        column_metadata = {}
        for col in df.columns:
            detected_type, reason = detect_problem_type(df[col])
            column_metadata[col] = {
                'detected_type': detected_type,
                'reason': reason
            }

        # Check if the request wants JSON (AJAX) or HTML (Form submission)
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return templates.TemplateResponse(
                request=request,
                name="select_target.html",
                context={
                    "file_path": path.replace("\\", "/"),
                    "overview": overview,
                    "column_metadata": column_metadata
                }
            )

        return UploadResponse(
            file_path=path,
            overview=overview,
            column_metadata={k: ColumnMetadata(**v) for k, v in column_metadata.items()}
        )
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
