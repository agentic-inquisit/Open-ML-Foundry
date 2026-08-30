import cv2
import numpy as np
import os
import tempfile
import hashlib
import secrets
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import FileResponse, Response
from fastapi import UploadFile, File
import sys
import os

# Ensure edge module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edge.jax_train import run_inference
from edge.preprocess import run_pipeline

app = FastAPI(title="LocalML finetune")

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prometheus Metrics ---
VIDEO_GEN_TOTAL = Counter("video_generation_total", "Total videos requested", ["status"])
VIDEO_GEN_SUCCESS = Counter("video_generation_success", "Total videos successfully generated")

# --- Auth Models & Mock DB ---
class User(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class VideoRequest(BaseModel):
    prompt: str

# Mock Database
MOCK_USERS = {}
ACTIVE_TOKENS = {}

def get_password_hash(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Auth Endpoints ---
@app.post("/auth/signup")
async def signup(user: User):
    if user.username in MOCK_USERS:
        raise HTTPException(status_code=400, detail="Username already exists")
    MOCK_USERS[user.username] = get_password_hash(user.password)
    return {"message": "User created successfully"}

@app.post("/auth/login", response_model=TokenResponse)
async def login(user: User):
    stored_hash = MOCK_USERS.get(user.username)
    if not stored_hash or stored_hash != get_password_hash(user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = secrets.token_hex(32)
    ACTIVE_TOKENS[token] = user.username
    return {"access_token": token}

@app.post("/generate-video")
async def generate_video(request: VideoRequest):
    """
    Generates a mock video based on the prompt.
    In a real scenario, this would call a Generative AI model like Sora, Runway, or Stable Video Diffusion.
    """
    VIDEO_GEN_TOTAL.labels(status="started").inc()
    try:
        # Create a temporary file for the video
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, "generated_video.mp4")
        
        # Video settings
        width, height = 640, 480
        fps = 24
        duration = 3 # seconds
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
        
        # Generate frames
        for i in range(fps * duration):
            # Create a dynamic background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add some color gradients or movements based on frame index
            c1 = (i * 2) % 255
            c2 = (i * 5) % 255
            frame[:,:] = [c1, 100, c2]
            
            # Overlay the prompt text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Prompt: {request.prompt}"
            cv2.putText(frame, text, (50, height // 2), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add some visual "generation" effect
            cv2.putText(frame, f"Generating frame {i}...", (50, height - 50), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            out.write(frame)
            
        out.release()
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Failed to create video file")
        
        VIDEO_GEN_TOTAL.labels(status="success").inc()
        VIDEO_GEN_SUCCESS.inc()
        return FileResponse(file_path, media_type="video/mp4", filename="stable_diffusion_video.mp4")
        
    except Exception as e:
        VIDEO_GEN_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

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

class ChatRequest(BaseModel):
    prompt: str
    model: str = "llama3" # wuth exec 'ollama pull llama3'

@app.post("/chat")
async def chat_with_llm(request: ChatRequest):
    try:
        # Connect to Ollama's local REST API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": request.model,
                "prompt": request.prompt,
                "stream": False # True for real-time word streaming
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "Vision Services is running", "endpoints": ["/generate-video", "/jax-inference", "/preprocess", "/chat"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
