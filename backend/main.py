import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import upload, train, analyze
from backend.utils import templates

from backend.ml.models import get_models
from backend.ml.trainer import detect_dataset_size
from backend.utils import read_csv_from_file

app = FastAPI(title="ML Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(PROJECT_ROOT, "frontend", "static")

# Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/available-models")
async def available_models(problem_type: str, file_path: str):
    try:
        df = read_csv_from_file(file_path)
        dataset_size = detect_dataset_size(df)
    except:
        dataset_size = "medium"
    
    models = get_models(problem_type, dataset_size)
    print(f"DEBUG: Problem Type: {problem_type}, Dataset Size: {dataset_size}")
    print(f"DEBUG: Available models: {list(models.keys())}")
    return {"models": list(models.keys())}

# Include Routers
app.include_router(upload.router, tags=["Upload"])
app.include_router(train.router, tags=["Training"])
app.include_router(analyze.router, tags=["Analysis"])

# View Routes (Preserving compatibility with existing frontend)
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
