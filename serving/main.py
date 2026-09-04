import os
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Ensure edge module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edge.jax_train import run_inference
from edge.preprocess import run_pipeline
from serving.session_api import router as session_router
from serving.features_api import router as features_router

app = FastAPI(title="Open ML Foundry")

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router)
app.include_router(features_router)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_chat_ui():
    """Session-based chat UI for LLM + vision fine-tuning."""
    with open(os.path.join(_STATIC_DIR, "sessions_chat.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/jax-inference")
async def jax_inference(file: UploadFile = File(...)):
    """
    Endpoint for JAX-based inference.
    """
    try:
        image_bytes = await file.read()
        result = run_inference(image_bytes)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess")
async def preprocess_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = run_pipeline(image_bytes)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "status": "Open ML Foundry is running",
        "endpoints": ["/sessions", "/api/v1/sessions", "/api/v1/models",
                      "/jax-inference", "/preprocess"],
    }

_ICON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "icon.ico"))

@app.get("/favicon.ico")
async def favicon():
    return FileResponse(_ICON_PATH, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
