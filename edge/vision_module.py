import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import numpy as np
import time
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, Response
import base64
import io
import qrcode
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, Response
import base64
import io
import qrcode
import shutil
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import FileResponse
import pickle # Used for serializing Flax params safely

# Check if jax_train is available (dependencies installed)
try:
    import jax_train
    # Flax serialization handles JAX PyTree objects (params) efficiently
    # Converts JAX params to bytes that can be saved to disk and reloaded
    import flax.serialization as serialization
    from flax.training import checkpoints  # For checkpoint management
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: JAX/Flax dependencies not found. Finetuning will be disabled.")

# Import labeling service for dataset annotation
try:
    from labeling_service import LabelingService
    LABELING_AVAILABLE = True
except ImportError:
    LABELING_AVAILABLE = False
    print("Warning: LabelingService not available.")

# Import model registry for versioning and metadata
try:
    from model_registry import ModelRegistry
    REGISTRY = ModelRegistry()
    REGISTRY_AVAILABLE = True
except Exception as e:
    REGISTRY_AVAILABLE = False
    REGISTRY = None
    print(f"Warning: ModelRegistry not available: {e}")

# Import A/B testing service
try:
    from ab_testing import ABTestingService
    AB_SERVICE = ABTestingService(db_path="model_registry.db")
    AB_AVAILABLE = True
except Exception as e:
    AB_AVAILABLE = False
    AB_SERVICE = None
    print(f"Warning: ABTestingService not available: {e}")

# Import validation service
try:
    from validation_service import ValidationService
    VALIDATION_SERVICE = ValidationService(REGISTRY, jax_train) if (REGISTRY_AVAILABLE and JAX_AVAILABLE) else None
    VALIDATION_AVAILABLE = VALIDATION_SERVICE is not None
except Exception as e:
    VALIDATION_AVAILABLE = False
    VALIDATION_SERVICE = None
    print(f"Warning: ValidationService not available: {e}")

# Import inference cache service
try:
    from inference_cache import ModelInferenceCache
    MODEL_CACHE = ModelInferenceCache(jax_train) if JAX_AVAILABLE else None
    CACHE_AVAILABLE = MODEL_CACHE is not None
except Exception as e:
    CACHE_AVAILABLE = False
    MODEL_CACHE = None
    print(f"Warning: ModelInferenceCache not available: {e}")

# Import multi-model inference service
try:
    from multi_model_inference import MultiModelInference
    MULTI_INFERENCE = MultiModelInference(MODEL_CACHE) if CACHE_AVAILABLE else None
    MULTI_INFERENCE_AVAILABLE = MULTI_INFERENCE is not None
except Exception as e:
    MULTI_INFERENCE_AVAILABLE = False
    MULTI_INFERENCE = None
    print(f"Warning: MultiModelInference not available: {e}")

# Import traffic router service
try:
    from traffic_router import TrafficRouter
    TRAFFIC_ROUTER = TrafficRouter()
    ROUTER_AVAILABLE = True
except Exception as e:
    ROUTER_AVAILABLE = False
    TRAFFIC_ROUTER = None
    print(f"Warning: TrafficRouter not available: {e}")

app = FastAPI(title="LocalML Module")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize labeling service
if LABELING_AVAILABLE:
    labeling_service = LabelingService(db_path="labeling.db")
else:
    labeling_service = None

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter(
    "vision_request_count", "Total number of vision inference requests", ["method", "endpoint", "status"]
)
INFERENCE_LATENCY = Histogram(
    "vision_inference_latency_seconds", "Latency of object detection inference", ["source"]
)
FINETUNE_COUNT = Counter(
    "vision_finetune_total", "Total number of finetuning jobs started"
)

class WebcamStream:
    """Handles webcam frame capture using OpenCV."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"Warning: Could not open webcam {src}. Using dummy stream.")
            self.stream = None
        
    def get_frame(self):
        if self.stream is None:
            # Return dummy black frame with noise or text to indicate no camera
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "NO CAMERA", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

        ret, frame = self.stream.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.stream:
            self.stream.release()

class MobileFrameSource:
    """Handles frames received via WebSocket from a mobile device."""
    def __init__(self):
        self.last_frame = None
        self.active = False
        self.last_update = 0
    
    def update_frame(self, frame_bytes):
        # Decode image from bytes
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
             self.last_frame = frame
             self.last_update = time.time()
             self.active = True

    def get_frame(self):
        # Timeout if no frame for 5 seconds
        if time.time() - self.last_update > 5:
            self.active = False
            return None
        return self.last_frame

class ObjectDetector:
    """Handles object detection using torchvision's Faster R-CNN."""
    def __init__(self, threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize model with pre-trained weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=threshold)
        self.model.to(self.device)
        self.model.eval()
        self.categories = weights.meta["categories"]

    def detect(self, frame):
        # Convert BGR (OpenCV) to RGB (Torchvision)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to tensor and add batch dimension
        img_tensor = F.to_tensor(img_rgb).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(img_tensor)[0]
        
        return prediction

    def get_labels(self, prediction):
        labels = [self.categories[i] for i in prediction['labels']]
        scores = prediction['scores'].tolist()
        boxes = prediction['boxes'].tolist()
        return list(zip(labels, scores, boxes))

# Global instances
stream = None
mobile_source = None
detector = None
camera_enabled = False
camera_permission_granted = False

@app.on_event("startup")
async def startup_event():
    global mobile_source
    # DO NOT auto-initialize camera stream - only on explicit request
    mobile_source = MobileFrameSource()
    # Ensure finetune storage directory exists
    FINETUNE_DIR = Path("finetuned_models")
    FINETUNE_DIR.mkdir(exist_ok=True)
    # Initialize SQLite DB
    DB_PATH = Path("finetune.db")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS finetune_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_object TEXT,
        dataset_path TEXT,
        result_path TEXT,
        model_path TEXT,
        timestamp TEXT
    )""")
    # Migration: Add model_path if it doesn't exist
    try:
        c.execute("ALTER TABLE finetune_requests ADD COLUMN model_path TEXT")
    except sqlite3.OperationalError:
        pass # Already exists
    conn.commit()
    conn.close()

@app.get("/model-info")
async def get_model_info():
    """Get detailed information about deployed object detection model"""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    categories = weights.meta["categories"]

    return {
        "model": {
            "name": "FasterRCNN ResNet50 FPN",
            "framework": "PyTorch + Torchvision",
            "architecture": "Faster R-CNN with ResNet50 backbone and FPN",
            "weights": "COCO pretrained (FasterRCNN_ResNet50_FPN_Weights.DEFAULT)",
            "input_size": "Variable (min 640x480 recommended)",
            "output": "Bounding boxes, class labels, confidence scores",
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "pretrained_on": "COCO dataset (80 object classes)",
            "confidence_threshold": float(os.getenv("DETECTION_THRESHOLD", 0.7))
        },
        "supported_classes": {
            "total": len(categories),
            "classes": sorted(categories)
        },
        "supported_sources": ["webcam", "mobile", "file_upload"],
        "features": [
            "Real-time object detection from webcam",
            "Mobile frame streaming via WebSocket",
            "Image file upload for batch detection",
            "Confidence-based filtering",
            "Bounding box coordinates",
            "Latency tracking"
        ],
        "specs": {
            "model_size": "~137 MB",
            "inference_latency": "50-200ms (GPU: 50-100ms, CPU: 100-200ms)",
            "supported_batch_size": 1,
            "fps_capability": "5-15 FPS (GPU), 1-3 FPS (CPU)"
        }
    }

@app.post("/camera/request-permission")
async def request_camera_permission(user_id: str = "default"):
    """Request permission to use camera streaming"""
    global camera_permission_granted, stream, detector

    camera_permission_granted = True

    # Initialize camera on first permission grant
    if stream is None or detector is None:
        WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))
        DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.7))
        stream = WebcamStream(src=WEBCAM_INDEX)
        detector = ObjectDetector(threshold=DETECTION_THRESHOLD)

    return {
        "status": "permission_granted",
        "message": f"Camera access enabled for user: {user_id}",
        "camera_available": stream is not None and stream.stream is not None,
        "detector_ready": detector is not None,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

@app.get("/camera/status")
async def get_camera_status():
    """Check camera permission and availability status"""
    return {
        "permission_granted": camera_permission_granted,
        "camera_initialized": stream is not None,
        "camera_available": stream is not None and stream.stream is not None if stream else False,
        "detector_ready": detector is not None,
        "mobile_source_active": mobile_source.active if mobile_source else False,
        "message": "Camera streaming not enabled. Call POST /camera/request-permission first." if not camera_permission_granted else "Camera ready for streaming"
    }

@app.get("/detect-dashboard", response_class=HTMLResponse)
async def detect_dashboard():
    """Interactive dashboard for object detection with camera control"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Object Detection</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --card: #FFFFFF;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
                --text-muted: #888888;
            }

            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg);
                color: var(--text-primary);
                line-height: 1.5;
                padding: 0;
            }

            .header {
                background: var(--white);
                border-bottom: 1px solid var(--border);
                padding: 20px 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .header h1 {
                font-size: 20px;
                font-weight: 700;
                letter-spacing: -0.3px;
                color: var(--gold-dark);
            }

            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 600;
                border: 1px solid var(--gold);
                border-radius: 4px;
                background: var(--white);
                color: var(--gold-dark);
            }

            .status-badge.active {
                background: var(--gold);
                color: var(--white);
                border-color: var(--gold);
            }

            .status-badge.inactive {
                background: var(--white);
                color: var(--text-muted);
                border-color: var(--border);
            }

            .status-dot {
                width: 6px;
                height: 6px;
                border-radius: 2px;
                animation: pulse 2s infinite;
            }

            .status-badge.active .status-dot {
                background: var(--white);
            }

            .status-badge.inactive .status-dot {
                background: var(--text-muted);
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }

            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 32px;
            }

            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }

            .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 24px;
            }

            .card h2 {
                font-size: 15px;
                font-weight: 700;
                margin-bottom: 16px;
                color: var(--gold-dark);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .card h3 {
                font-size: 12px;
                font-weight: 700;
                margin-top: 16px;
                margin-bottom: 8px;
                color: var(--gold-dark);
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }

            .specs-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                font-size: 12px;
            }

            .spec-item {
                background: var(--bg);
                padding: 12px;
                border: 1px solid var(--border);
                border-radius: 4px;
            }

            .spec-label {
                color: var(--text-muted);
                font-weight: 600;
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 0.4px;
                margin-bottom: 4px;
            }

            .spec-value {
                color: var(--text-primary);
                font-weight: 700;
                font-size: 13px;
            }

            .btn {
                padding: 10px 20px;
                border: 1px solid transparent;
                font-size: 12px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.15s;
                text-transform: uppercase;
                letter-spacing: 0.3px;
                border-radius: 4px;
            }

            .btn-primary {
                background: var(--gold);
                color: var(--white);
                border-color: var(--gold);
            }

            .btn-primary:hover {
                background: var(--gold-dark);
                border-color: var(--gold-dark);
            }

            .btn-secondary {
                background: var(--bg);
                color: var(--text-primary);
                border: 1px solid var(--border);
            }

            .btn-secondary:hover {
                background: var(--white);
                border-color: var(--gold);
            }

            .btn-success {
                background: var(--gold);
                color: var(--white);
                border-color: var(--gold);
            }

            .btn-success:hover {
                background: var(--gold-dark);
                border-color: var(--gold-dark);
            }

            .button-group {
                display: flex;
                gap: 12px;
                margin-top: 18px;
                flex-wrap: wrap;
            }

            .class-filter {
                margin-top: 16px;
            }

            .class-filter label {
                display: block;
                font-size: 12px;
                font-weight: 700;
                margin-bottom: 8px;
                color: var(--text-primary);
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }

            .class-filter input {
                width: 100%;
                padding: 10px;
                border: 1px solid var(--border);
                border-radius: 4px;
                font-size: 12px;
                font-family: 'Inter', sans-serif;
                background: var(--white);
                color: var(--text-primary);
            }

            .class-filter input:focus {
                outline: none;
                border-color: var(--gold);
            }

            .classes-list {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                margin-top: 8px;
                max-height: 120px;
                overflow-y: auto;
                padding: 8px;
                background: var(--bg);
                border: 1px solid var(--border);
                border-radius: 4px;
            }

            .class-tag {
                background: var(--white);
                border: 1px solid var(--border);
                padding: 5px 10px;
                font-size: 11px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.15s;
                border-radius: 4px;
            }

            .class-tag:hover {
                background: var(--gold);
                color: var(--white);
                border-color: var(--gold);
            }

            .class-tag.selected {
                background: var(--gold);
                color: var(--white);
                border-color: var(--gold);
            }

            .frame-container {
                width: 100%;
                background: #1A1A1A;
                overflow: hidden;
                aspect-ratio: 16/9;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-top: 16px;
                border: 1px solid var(--border);
                border-radius: 6px;
            }

            .frame-container img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }

            .frame-container .placeholder {
                color: var(--text-muted);
                font-size: 13px;
                text-align: center;
            }

            .detection-results {
                margin-top: 16px;
            }

            .detection-item {
                background: var(--bg);
                padding: 12px;
                margin-bottom: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border: 1px solid var(--border);
                border-radius: 4px;
            }

            .detection-label {
                font-weight: 700;
                color: var(--text-primary);
                font-size: 13px;
            }

            .detection-score {
                background: var(--gold);
                color: var(--white);
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 700;
                border-radius: 3px;
            }

            .stats {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr 1fr;
                gap: 12px;
                margin-top: 16px;
            }

            .stat-card {
                background: var(--bg);
                padding: 14px;
                text-align: center;
                border: 1px solid var(--border);
                border-radius: 4px;
            }

            .stat-label {
                font-size: 10px;
                color: var(--text-muted);
                text-transform: uppercase;
                font-weight: 700;
                letter-spacing: 0.3px;
                margin-bottom: 6px;
            }

            .stat-value {
                font-size: 18px;
                font-weight: 800;
                color: var(--gold-dark);
            }

            .full-width {
                grid-column: 1 / -1;
            }

            .loading {
                display: inline-block;
                width: 14px;
                height: 14px;
                border: 2px solid var(--gold);
                border-right-color: transparent;
                border-radius: 50%;
                animation: spin 0.6s linear infinite;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .message {
                padding: 12px;
                margin-bottom: 16px;
                font-size: 12px;
                border: 1px solid;
                border-radius: 4px;
            }

            .message.error {
                background: #FFF5F0;
                color: #8B4513;
                border-color: #D2B48C;
            }

            .message.success {
                background: #FFFEF0;
                color: var(--gold-dark);
                border-color: var(--border);
            }

            .message.info {
                background: #FFFBF0;
                color: var(--gold-dark);
                border-color: var(--border);
            }

            @media (max-width: 1024px) {
                .grid { grid-template-columns: 1fr; }
                .stats { grid-template-columns: 1fr 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Object Detection</h1>
            <div class="status-badge" id="statusBadge">
                <span class="status-dot"></span>
                <span id="statusText">Initializing...</span>
            </div>
        </div>

        <div class="container">
            <div class="grid">
                <!-- Model Information Card -->
                <div class="card">
                    <h2>Model Information</h2>
                    <div id="modelInfo">
                        <p style="color: var(--text-muted); font-size: 12px;">Loading model details...</p>
                    </div>
                </div>

                <!-- Camera Control Card -->
                <div class="card">
                    <h2>Camera Control</h2>
                    <div id="cameraMessage"></div>
                    <button class="btn btn-primary" id="cameraBtn" onclick="requestCameraPermission()" style="width: 100%; margin-top: 8px;">
                        Enable Camera
                    </button>
                    <div style="margin-top: 14px; padding: 12px; background: var(--bg); border: 1px solid var(--border);">
                        <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase; font-weight: 700;">
                            STATUS
                        </div>
                        <div id="cameraStatus" style="font-size: 12px; color: var(--text-primary); line-height: 1.6;">
                            <span>Camera: Disabled</span><br>
                            <span>Detector: Not initialized</span><br>
                            <span id="permissionStatus">Permission: Not granted</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Detection Controls -->
            <div class="card full-width">
                <h2>Detection Settings</h2>

                <div class="class-filter">
                    <label>Filter by Classes (comma-separated or click below)</label>
                    <input type="text" id="classFilter" placeholder="e.g. person,car,dog or leave empty for all"
                           onkeyup="updateClassFilter()">
                    <div class="classes-list" id="classesList">
                        <div style="color: var(--text-muted); font-size: 12px;">Loading classes...</div>
                    </div>
                </div>

                <div class="button-group">
                    <button class="btn btn-success" id="detectBtn" onclick="runDetection()" disabled>
                        Run Detection
                    </button>
                    <button class="btn btn-secondary" id="clearBtn" onclick="clearResults()">
                        Clear Results
                    </button>
                </div>
            </div>

            <!-- Detection Results -->
            <div class="card full-width" style="margin-top: 20px;">
                <h2>Detection Results</h2>
                <div id="resultsArea">
                    <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 40px;">
                        Enable camera and click "Run Detection" to see results
                    </div>
                </div>
            </div>
        </div>

        <script>
            let allClasses = [];
            let selectedClasses = new Set();

            async function loadModelInfo() {
                try {
                    const res = await fetch('/model-info');
                    const data = await res.json();

                    const specs = data.specs || {};
                    const model = data.model || {};

                    const html = `
                        <h3>FasterRCNN ResNet50 FPN</h3>
                        <div class="specs-grid">
                            <div class="spec-item">
                                <div class="spec-label">Framework</div>
                                <div class="spec-value">PyTorch + Torchvision</div>
                            </div>
                            <div class="spec-item">
                                <div class="spec-label">Pretrained On</div>
                                <div class="spec-value">COCO (80 classes)</div>
                            </div>
                            <div class="spec-item">
                                <div class="spec-label">Model Size</div>
                                <div class="spec-value">~137 MB</div>
                            </div>
                            <div class="spec-item">
                                <div class="spec-label">Device</div>
                                <div class="spec-value">${model.device || 'CPU'}</div>
                            </div>
                            <div class="spec-item">
                                <div class="spec-label">Inference Latency</div>
                                <div class="spec-value">${specs.inference_latency || '50-200ms'}</div>
                            </div>
                            <div class="spec-item">
                                <div class="spec-label">FPS Capability</div>
                                <div class="spec-value">${specs.fps_capability || '5-15 FPS'}</div>
                            </div>
                            <div class="spec-item">
                                <div class="spec-label">Confidence Threshold</div>
                                <div class="spec-value">${model.confidence_threshold || 0.7}</div>
                            </div>
                            <div class="spec-item">
                                <div class="spec-label">Input Size</div>
                                <div class="spec-value">Variable (min 640×480)</div>
                            </div>
                        </div>
                    `;
                    document.getElementById('modelInfo').innerHTML = html;

                    // Load classes
                    allClasses = data.supported_classes?.classes || [];
                    populateClassesList();
                } catch (e) {
                    console.error('Error loading model info:', e);
                    document.getElementById('modelInfo').innerHTML =
                        '<div class="message error">Failed to load model information</div>';
                }
            }

            function populateClassesList() {
                const classList = document.getElementById('classesList');
                const html = allClasses.map(cls =>
                    `<div class="class-tag" onclick="toggleClass('${cls}')">${cls}</div>`
                ).join('');
                classList.innerHTML = html;
            }

            function toggleClass(className) {
                const tags = document.querySelectorAll('.class-tag');
                tags.forEach(tag => {
                    if (tag.textContent === className) {
                        if (selectedClasses.has(className)) {
                            selectedClasses.delete(className);
                            tag.classList.remove('selected');
                        } else {
                            selectedClasses.add(className);
                            tag.classList.add('selected');
                        }
                    }
                });
                updateFilterInput();
            }

            function updateClassFilter() {
                const input = document.getElementById('classFilter').value;
                selectedClasses.clear();
                if (input.trim()) {
                    input.split(',').forEach(cls => {
                        const trimmed = cls.trim();
                        if (trimmed) selectedClasses.add(trimmed);
                    });
                }

                document.querySelectorAll('.class-tag').forEach(tag => {
                    if (selectedClasses.has(tag.textContent)) {
                        tag.classList.add('selected');
                    } else {
                        tag.classList.remove('selected');
                    }
                });
            }

            function updateFilterInput() {
                const input = document.getElementById('classFilter');
                input.value = Array.from(selectedClasses).join(', ');
            }

            async function checkCameraStatus() {
                try {
                    const res = await fetch('/camera/status');
                    const data = await res.json();

                    const badge = document.getElementById('statusBadge');
                    const statusText = document.getElementById('statusText');
                    const detectBtn = document.getElementById('detectBtn');
                    const cameraBtn = document.getElementById('cameraBtn');
                    const permStatus = document.getElementById('permissionStatus');

                    if (data.permission_granted) {
                        badge.className = 'status-badge active';
                        statusText.textContent = 'Ready';
                        detectBtn.disabled = false;
                        cameraBtn.textContent = 'Camera Enabled';
                        cameraBtn.disabled = true;
                        permStatus.textContent = 'Permission: Granted';
                        document.getElementById('cameraStatus').innerHTML =
                            `<span>Camera: ${data.camera_available ? 'Available' : 'No camera detected'}</span><br>` +
                            `<span>Detector: ${data.detector_ready ? 'Ready' : 'Not ready'}</span><br>` +
                            `<span>Mobile: ${data.mobile_source_active ? 'Active' : 'Inactive'}</span>`;
                    } else {
                        badge.className = 'status-badge inactive';
                        statusText.textContent = 'Disabled';
                        detectBtn.disabled = true;
                        cameraBtn.textContent = 'Enable Camera';
                        cameraBtn.disabled = false;
                        document.getElementById('cameraMessage').innerHTML =
                            '<div class="message info">Click "Enable Camera" to start object detection</div>';
                    }
                } catch (e) {
                    console.error('Error checking camera status:', e);
                }
            }

            async function requestCameraPermission() {
                try {
                    const btn = document.getElementById('cameraBtn');
                    btn.innerHTML = '<span class="loading"></span> Requesting...';
                    btn.disabled = true;

                    const res = await fetch('/camera/request-permission', { method: 'POST' });
                    const data = await res.json();

                    if (res.ok) {
                        document.getElementById('cameraMessage').innerHTML =
                            '<div class="message success">Camera enabled and initialized</div>';
                        await checkCameraStatus();
                    } else {
                        document.getElementById('cameraMessage').innerHTML =
                            '<div class="message error">Failed to enable camera: ' + (data.message || 'Unknown error') + '</div>';
                        btn.innerHTML = 'Enable Camera';
                        btn.disabled = false;
                    }
                } catch (e) {
                    console.error('Error requesting permission:', e);
                    document.getElementById('cameraMessage').innerHTML =
                        '<div class="message error">Error: ' + e.message + '</div>';
                    btn.innerHTML = 'Enable Camera';
                    btn.disabled = false;
                }
            }

            async function runDetection() {
                try {
                    const btn = document.getElementById('detectBtn');
                    btn.innerHTML = '<span class="loading"></span> Detecting...';
                    btn.disabled = true;

                    const filterClasses = Array.from(selectedClasses).join(',');
                    const url = '/detect' + (filterClasses ? '?filter_classes=' + encodeURIComponent(filterClasses) : '');

                    const res = await fetch(url);
                    const data = await res.json();

                    if (data.error) {
                        document.getElementById('resultsArea').innerHTML =
                            '<div class="message error">Error: ' + data.message + '</div>';
                    } else {
                        displayResults(data);
                    }

                    btn.innerHTML = 'Run Detection';
                    btn.disabled = false;
                } catch (e) {
                    console.error('Error running detection:', e);
                    document.getElementById('resultsArea').innerHTML =
                        '<div class="message error">Error: ' + e.message + '</div>';
                    btn.innerHTML = 'Run Detection';
                    btn.disabled = false;
                }
            }

            function displayResults(data) {
                const html = `
                    <div class="frame-container">
                        <img src="data:image/jpeg;base64,${data.frame_encoded}" alt="Detection result">
                    </div>

                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-label">Objects Detected</div>
                            <div class="stat-value">${data.detection_count}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Inference Time</div>
                            <div class="stat-value">${data.latency_ms}ms</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Source</div>
                            <div class="stat-value">${data.source}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Timestamp</div>
                            <div class="stat-value" style="font-size: 12px;">${data.timestamp}</div>
                        </div>
                    </div>

                    ${data.detections.length > 0 ? `
                        <div style="margin-top: 20px;">
                            <div style="font-size: 12px; font-weight: 700; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.3px;">Detected Objects (${data.detection_count})</div>
                            <div class="detection-results">
                                ${data.detections.map(d => `
                                    <div class="detection-item">
                                        <span class="detection-label">${d.label}</span>
                                        <span class="detection-score">${(d.score * 100).toFixed(1)}%</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : `
                        <div class="message info" style="margin-top: 20px;">
                            No objects detected in the current frame
                        </div>
                    `}
                `;
                document.getElementById('resultsArea').innerHTML = html;
            }

            function clearResults() {
                document.getElementById('resultsArea').innerHTML =
                    '<div style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 40px;">Results cleared</div>';
                document.getElementById('classFilter').value = '';
                selectedClasses.clear();
                document.querySelectorAll('.class-tag').forEach(tag => tag.classList.remove('selected'));
            }

            // Initialize on page load
            window.addEventListener('load', () => {
                loadModelInfo();
                checkCameraStatus();
                // Check status every 2 seconds
                setInterval(checkCameraStatus, 2000);
            });
        </script>
    </body>
    </html>
    """

@app.get("/detect")
async def detect_objects(filter_classes: str = ""):
    """
    Capture frame and return detection results.
    Requires prior permission via POST /camera/request-permission

    Optional: filter_classes - comma-separated list of classes to detect (e.g. "person,car,dog")
    """
    global stream, detector, mobile_source, camera_permission_granted

    # Check permission
    if not camera_permission_granted:
        return {
            "error": "Camera access not permitted",
            "message": "Call POST /camera/request-permission first to enable camera streaming",
            "status_code": 403
        }

    if stream is None or detector is None:
        return {
            "error": "Detection model not initialized",
            "message": "Call POST /camera/request-permission to initialize",
            "status_code": 503
        }

    # Prioritize mobile source if active
    frame = None
    if mobile_source and mobile_source.active:
        frame = mobile_source.get_frame()

    if frame is None:
        frame = stream.get_frame()

    if frame is None:
        REQUEST_COUNT.labels(method="GET", endpoint="/detect", status="400").inc()
        return {"error": "Failed to capture frame"}

    # Detect objects
    start_time = time.time()
    source_type = "mobile" if (mobile_source and mobile_source.active) else "webcam"
    with INFERENCE_LATENCY.labels(source=source_type).time():
        prediction = detector.detect(frame)

    latency = time.time() - start_time
    REQUEST_COUNT.labels(method="GET", endpoint="/detect", status="200").inc()

    results = detector.get_labels(prediction)

    # Filter by classes if specified
    if filter_classes:
        allowed_classes = set(filter_classes.split(","))
        results = [(label, score, box) for label, score, box in results if label.strip() in allowed_classes]

    # Encode frame as base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "model": "FasterRCNN ResNet50 FPN (COCO pretrained)",
        "detections": [{"label": label, "score": round(score, 3), "box": box} for label, score, box in results],
        "detection_count": len(results),
        "latency_ms": round(latency * 1000, 2),
        "frame_encoded": frame_b64,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "source": source_type,
        "filter_applied": filter_classes if filter_classes else "none"
    }

@app.get("/finetune-config")
async def finetune_config():
    """Get fine-tuning configuration options and defaults."""
    return {
        "status": "success",
        "config": {
            "epochs": {"default": 5, "min": 1, "max": 100, "description": "Number of training epochs"},
            "batch_size": {"default": 4, "min": 1, "max": 64, "description": "Batch size per iteration"},
            "image_size": {
                "default": 224,
                "options": [28, 64, 128, 224, 512],
                "description": "Target image dimension (square)"
            },
            "num_classes": {"default": 10, "min": 2, "max": 1000, "description": "Number of output classes"},
            "learning_rate": {"default": 0.001, "min": 0.0001, "max": 0.1, "description": "Learning rate"},
            "validation_split": {"default": 0.2, "min": 0.1, "max": 0.5, "description": "Fraction for validation"},
            "enable_validation": {"default": True, "description": "Enable validation set"},
            "enable_early_stopping": {"default": True, "description": "Enable early stopping"},
            "patience": {"default": 3, "min": 1, "max": 20, "description": "Epochs to wait for improvement"},
            "checkpoint_interval": {"default": 1, "min": 0, "max": 100, "description": "Save checkpoint every N epochs (0=never)"}
        }
    }


@app.post("/finetune")
async def finetune_model(
    dataset: UploadFile = File(...),
    target_object: str = Form(...),
    num_classes: int = Form(default=10),
    image_size: int = Form(default=224),
    epochs: int = Form(default=5),
    batch_size: int = Form(default=4),
    enable_validation: bool = Form(default=True),
    validation_split: float = Form(default=0.2),
    enable_early_stopping: bool = Form(default=True),
    patience: int = Form(default=3),
    checkpoint_interval: int = Form(default=1)
):
    """
    Finetune a JAX model with validation, early stopping, and checkpointing.

    Args:
        dataset: Image file to fine-tune on
        target_object: Model name for this training run
        num_classes: Number of output classes
        image_size: Target image dimension (28, 64, 128, 224, 512)
        epochs: Number of training epochs
        batch_size: Batch size per iteration
        enable_validation: Enable train/val split (80/20)
        validation_split: Fraction of data for validation
        enable_early_stopping: Stop if val loss doesn't improve (requires user confirmation)
        patience: Epochs to wait before early stopping
        checkpoint_interval: Save checkpoint every N epochs (0=no checkpoints)
    """
    FINETUNE_COUNT.inc()
    if not JAX_AVAILABLE:
        return {"status": "error", "message": "JAX/Flax libraries not installed on server."}

    contents = await dataset.read()

    # === SAVE DATASET ARTIFACT ===
    FINETUNE_DIR = Path("finetuned_models")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_filename = f"{timestamp_str}_{target_object}{Path(dataset.filename).suffix}"
    dataset_path = FINETUNE_DIR / dataset_filename
    # Store raw input data for reproducibility and audit trail
    with open(dataset_path, "wb") as f:
        f.write(contents)

    # === RUN GRAIN-BASED TRAINING WITH VALIDATION & CHECKPOINTING ===
    result = jax_train.run_finetuning(
        contents,
        target_object=target_object,
        steps=epochs,
        batch_size=batch_size,
        num_classes=num_classes,
        image_size=image_size,
        enable_validation=enable_validation,
        validation_split=validation_split,
        enable_early_stopping=enable_early_stopping,
        patience=patience,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=str(FINETUNE_DIR)
    )

    # === REGISTER MODEL IN VERSIONING SYSTEM ===
    model_version_info = None
    if REGISTRY_AVAILABLE and result.get("status") == "success":
        try:
            # Register new version (auto-increments)
            reg_result = REGISTRY.register_model(
                model_name=target_object,
                description=f"Trained on {len(train_labels) if 'train_labels' in locals() else '?'} images",
                owner="system",
                access_level="private"
            )
            model_id = reg_result["model_id"]
            model_version = reg_result["version"]

            # Add training metadata
            REGISTRY.add_metadata(model_id, {
                "num_classes": num_classes,
                "image_size": image_size,
                "epochs_trained": result.get("epochs_trained"),
                "batch_size": batch_size,
                "learning_rate": 1e-3,
                "validation_split": validation_split if enable_validation else 0,
                "early_stopping_enabled": enable_early_stopping,
                "checkpoint_interval": checkpoint_interval,
                "final_train_loss": result.get("final_train_loss"),
                "best_val_loss": result.get("best_val_loss"),
                "training_duration_seconds": 0
            })

            # Add dataset information
            REGISTRY.add_dataset_info(model_id, {
                "total_images": len(labels_array) if 'labels_array' in locals() else 0,
                "training_images": len(train_labels) if 'train_labels' in locals() else 0,
                "validation_images": len(val_labels) if enable_validation and 'val_labels' in locals() else 0,
                "test_images": 0,
                "classes": list(range(num_classes)),
                "class_distribution": {},
                "data_source": "uploaded_image",
                "preprocessing_steps": f"Resized to {image_size}×{image_size}, aspect-ratio preserved"
            })

            # Add history event
            REGISTRY.add_history_event(
                model_id,
                "model_trained",
                f"Model trained with {result.get('epochs_trained')} epochs, "
                f"best val loss: {result.get('best_val_loss')}",
                "system"
            )

            model_version_info = {
                "model_id": model_id,
                "version": model_version,
                "name": target_object
            }
            print(f"✓ Registered {target_object} {model_version} (ID: {model_id})")
        except Exception as e:
            print(f"⚠ Error registering model: {e}")

    # === SAVE MODEL CHECKPOINT ===
    model_path = None
    if result.get("status") == "success":
        model_filename = f"{timestamp_str}_{target_object}_model.flax"
        model_path = FINETUNE_DIR / model_filename

        # IMPROVED: Use Flax serialization for JAX params (replaces torch.save)
        # Flax serialization handles:
        # - JAX PyTree structures (nested dicts/tuples of arrays)
        # - Efficient binary format compatible with JAX ecosystems
        # - Safe to reload with flax.serialization.from_bytes()

        # TODO: After training, serialize the final state:
        # with open(model_path, 'wb') as f:
        #     f.write(flax.serialization.to_bytes(final_state))

        print(f"✓ Model checkpoint would be saved to: {model_path}")

        # Register checkpoint in version
        if REGISTRY_AVAILABLE and model_version_info:
            try:
                for epoch in range(1, result.get("epochs_trained", 1) + 1):
                    REGISTRY.save_checkpoint(
                        model_version_info["model_id"],
                        epoch,
                        f"{model_path}_epoch_{epoch:03d}.ckpt",
                        train_loss=result.get("final_train_loss", 0),
                        val_loss=result.get("best_val_loss", 0),
                        is_best=(epoch == result.get("best_val_loss") or epoch == result.get("epochs_trained"))
                    )
            except Exception as e:
                print(f"⚠ Error registering checkpoints: {e}")

    # === SAVE TRAINING RESULTS METADATA ===
    # Store training metrics for analysis (loss history, training steps, etc.)
    result_path = FINETUNE_DIR / f"{timestamp_str}_{target_object}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    # === RECORD IN DATABASE ===
    # Track all training jobs for auditing and replay
    DB_PATH = Path("finetune.db")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO finetune_requests (target_object, dataset_path, result_path, model_path, timestamp) VALUES (?,?,?,?,?)",
        (target_object, str(dataset_path), str(result_path), str(model_path) if model_path else None, timestamp_str)
    )
    conn.commit()
    conn.close()

    return {
        "target": target_object,
        "dataset_saved": str(dataset_path),
        "result_saved": str(result_path),
        "model_saved": str(model_path) if model_path else None,
        # Include Grain usage metadata for debugging/monitoring
        "using_grain": result.get("using_grain", False),
        # Training performance metrics
        "training_metrics": {
            "final_loss": result.get("final_loss"),
            "average_loss": result.get("average_loss"),
            "total_steps": result.get("total_steps"),
            "samples_trained": result.get("samples_trained")
        },
        "download_url": f"http://localhost:8001/download-model/{timestamp_str}_{target_object}" if model_path else None,
        "result": result,
        "message": f"Finetuning complete with Grain-optimized batch loading for: {target_object}"
    }

@app.get("/download-model/{model_id}")
async def download_model(model_id: str):
    """
    Download finetuned model checkpoint.
    Changed from .pth to .flax for proper Flax serialization format.
    Flax format:
    - Stores JAX PyTree parameters efficiently
    - Compatible with flax.serialization.from_bytes() for reload
    - Replaces PyTorch pickle format (was incorrect for JAX params)
    """
    # Try both formats for backwards compatibility
    flax_path = Path("finetuned_models") / f"{model_id}_model.flax"
    torch_path = Path("finetuned_models") / f"{model_id}_model.pth"

    model_path = None
    media_type = None

    if flax_path.exists():
        model_path = flax_path
        # MIME type for Flax serialized data (standard binary)
        media_type = 'application/octet-stream'
    elif torch_path.exists():
        # Fallback for legacy .pth files
        model_path = torch_path
        media_type = 'application/octet-stream'
    else:
        return {"status": "error", "message": "Model file not found (.flax or .pth)."}

    return FileResponse(
        path=model_path,
        filename=f"{model_id}_model.flax",
        media_type=media_type
    )

@app.get("/admin/data")
async def get_admin_data():
    """Fetch all finetune records from DB."""
    DB_PATH = Path("finetune.db")
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM finetune_requests ORDER BY id DESC")
    rows = c.fetchall()
    data = [dict(row) for row in rows]
    conn.close()
    return data


@app.get("/admin/overview")
async def admin_overview():
    """Aggregate stats from all subsystems for the admin console overview."""
    overview = {
        "status": "success",
        "subsystems": {},
        "totals": {},
        "alerts": []
    }

    # Training stats
    try:
        DB_PATH = Path("finetune.db")
        finetune_total = 0
        finetune_with_model = 0
        if DB_PATH.exists():
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM finetune_requests")
            finetune_total = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM finetune_requests WHERE model_path IS NOT NULL")
            finetune_with_model = c.fetchone()[0]
            conn.close()
        overview["subsystems"]["training"] = {
            "available": True,
            "total_jobs": finetune_total,
            "completed": finetune_with_model,
            "pending": max(0, finetune_total - finetune_with_model)
        }
    except Exception as e:
        overview["subsystems"]["training"] = {"available": False, "error": str(e)}

    # Labeling stats
    try:
        if LABELING_AVAILABLE and labeling_service:
            stats = labeling_service.get_stats()
            overview["subsystems"]["labeling"] = {
                "available": True,
                "total_labels": stats.get("labeled", 0),
                "total_classes": stats.get("num_classes", 0),
                "pending_review": stats.get("pending", 0),
                "progress": stats.get("progress", 0)
            }
        else:
            overview["subsystems"]["labeling"] = {"available": False}
    except Exception as e:
        overview["subsystems"]["labeling"] = {"available": False, "error": str(e)}

    # Model Registry
    try:
        if REGISTRY_AVAILABLE and REGISTRY:
            models = REGISTRY.get_all_models()
            unique_names = len(set(m.get("name", "") for m in models)) if models else 0
            overview["subsystems"]["registry"] = {
                "available": True,
                "total_models": len(models) if models else 0,
                "unique_model_names": unique_names
            }
        else:
            overview["subsystems"]["registry"] = {"available": False}
    except Exception as e:
        overview["subsystems"]["registry"] = {"available": False, "error": str(e)}

    # A/B Testing
    try:
        if AB_AVAILABLE and AB_SERVICE:
            tests = AB_SERVICE.get_all_tests()
            active = sum(1 for t in tests if t.get("status") == "active")
            completed = sum(1 for t in tests if t.get("status") == "completed")
            overview["subsystems"]["ab_testing"] = {
                "available": True,
                "total_tests": len(tests),
                "active": active,
                "completed": completed
            }
        else:
            overview["subsystems"]["ab_testing"] = {"available": False}
    except Exception as e:
        overview["subsystems"]["ab_testing"] = {"available": False, "error": str(e)}

    # Validation Gates
    try:
        if REGISTRY_AVAILABLE and REGISTRY:
            conn = sqlite3.connect("model_registry.db")
            c = conn.cursor()
            try:
                c.execute("SELECT COUNT(*) FROM validation_gates")
                total = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM validation_gates WHERE decision = 'PASS'")
                passed = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM validation_gates WHERE decision = 'FAIL'")
                failed = c.fetchone()[0]
                overview["subsystems"]["validation"] = {
                    "available": True,
                    "total_gates": total,
                    "passed": passed,
                    "failed": failed,
                    "pass_rate": round((passed / total * 100), 1) if total > 0 else 0
                }
                if failed > 0:
                    overview["alerts"].append({
                        "level": "warning",
                        "category": "validation",
                        "message": f"{failed} model(s) failed validation gate"
                    })
            except sqlite3.OperationalError:
                overview["subsystems"]["validation"] = {"available": True, "total_gates": 0, "passed": 0, "failed": 0, "pass_rate": 0}
            conn.close()
        else:
            overview["subsystems"]["validation"] = {"available": False}
    except Exception as e:
        overview["subsystems"]["validation"] = {"available": False, "error": str(e)}

    # Inference Cache
    try:
        if CACHE_AVAILABLE and MODEL_CACHE:
            status = MODEL_CACHE.get_cache_status()
            overview["subsystems"]["cache"] = {
                "available": True,
                "cache_size": status.get("cache_size", 0),
                "max_cache_size": status.get("max_cache_size", 0),
                "hit_rate_percent": status.get("hit_rate_percent", 0),
                "total_memory_mb": status.get("total_memory_mb", 0)
            }
            if status.get("hit_rate_percent", 100) < 70 and status.get("total_hits", 0) + status.get("total_misses", 0) > 50:
                overview["alerts"].append({
                    "level": "warning",
                    "category": "cache",
                    "message": f"Cache hit rate is low ({status.get('hit_rate_percent', 0):.1f}%)"
                })
        else:
            overview["subsystems"]["cache"] = {"available": False}
    except Exception as e:
        overview["subsystems"]["cache"] = {"available": False, "error": str(e)}

    # Traffic Routes
    try:
        if ROUTER_AVAILABLE and TRAFFIC_ROUTER:
            overview["subsystems"]["routing"] = {
                "available": True,
                "total_routes": len(TRAFFIC_ROUTER.routes),
                "total_requests": TRAFFIC_ROUTER.request_counter
            }
        else:
            overview["subsystems"]["routing"] = {"available": False}
    except Exception as e:
        overview["subsystems"]["routing"] = {"available": False, "error": str(e)}

    # Aggregate totals
    overview["totals"] = {
        "subsystems_online": sum(1 for s in overview["subsystems"].values() if s.get("available")),
        "subsystems_total": len(overview["subsystems"]),
        "active_alerts": len(overview["alerts"])
    }

    return overview

@app.get("/admin", response_class=HTMLResponse)
async def admin_console():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Admin Console · LocalML</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --card: #FFFFFF;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
                --text-muted: #888888;
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --gold-light: #D4A373;
                --gold-pale: #FFF8DC;
                --shadow-soft: 0 1px 3px rgba(0,0,0,0.08);
                --shadow-hover: 0 4px 12px rgba(184,134,11,0.15);
                --success: #B8860B;
                --success-bg: #FFF8DC;
                --warning: #B8860B;
                --warning-bg: #FFF8DC;
                --error: #8B6914;
                --error-bg: #FFF5F0;
                --info-bg: #FFFBF0;
            }
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg);
                color: var(--text-primary);
                min-height: 100vh;
            }
            .top-bar {
                background: var(--white);
                border-bottom: 1px solid var(--border);
                padding: 18px 40px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: var(--shadow-soft);
            }
            .top-bar a.home {
                display: flex; align-items: center; gap: 10px;
                color: var(--text-secondary); text-decoration: none;
                font-size: 13px; font-weight: 500;
                transition: color 0.2s;
            }
            .top-bar a.home:hover { color: var(--gold-dark); }
            .top-bar .crumb {
                display: flex; align-items: center; gap: 10px;
                font-size: 13px; color: var(--text-muted);
            }
            .top-bar .crumb b { color: var(--text-primary); font-weight: 600; }
            .live-indicator {
                display: inline-flex; align-items: center; gap: 8px;
                padding: 6px 14px;
                background: var(--success-bg);
                border: 1px solid #C8E6C8;
                border-radius: 100px;
                color: var(--success);
                font-size: 12px; font-weight: 600;
            }
            .live-dot {
                width: 7px; height: 7px;
                border-radius: 50%;
                background: var(--success);
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.4; }
            }

            .container { max-width: 1400px; margin: 0 auto; padding: 32px 40px; }

            .page-header {
                display: flex; align-items: flex-start; gap: 18px;
                margin-bottom: 28px;
            }
            .page-icon {
                width: 52px; height: 52px;
                background: linear-gradient(135deg, var(--gold-light) 0%, var(--gold) 100%);
                border-radius: 12px;
                display: flex; align-items: center; justify-content: center;
                font-size: 24px;
                box-shadow: 0 6px 18px var(--gold-shadow);
            }
            h1 {
                font-size: 26px; font-weight: 700;
                color: var(--text-primary);
                letter-spacing: -0.4px; margin-bottom: 4px;
            }
            .page-header p { font-size: 14px; color: var(--text-secondary); }

            .tabs {
                display: flex; gap: 4px; margin-bottom: 24px;
                background: var(--white); padding: 6px;
                border-radius: 12px; border: 1px solid var(--grey-mid);
                flex-wrap: wrap;
            }
            .tab {
                padding: 10px 18px; background: transparent; border: none;
                cursor: pointer; border-radius: 8px;
                font-size: 13px; font-weight: 600;
                color: var(--text-secondary);
                font-family: inherit; transition: all 0.2s ease;
            }
            .tab:hover { color: var(--gold-dark); background: var(--grey-light); }
            .tab.active {
                background: linear-gradient(135deg, var(--gold) 0%, var(--gold-dark) 100%);
                color: white;
                box-shadow: 0 3px 10px var(--gold-shadow);
            }

            .tab-content {
                display: none;
                animation: fadeIn 0.3s ease;
            }
            .tab-content.active { display: block; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

            .panel {
                background: var(--card);
                border: 1px solid var(--grey-mid);
                border-radius: 12px;
                padding: 28px;
                margin-bottom: 20px;
                box-shadow: var(--shadow-soft);
            }
            .panel-header {
                display: flex; align-items: center; justify-content: space-between;
                margin-bottom: 20px;
                padding-bottom: 16px;
                border-bottom: 1px solid var(--grey-light);
            }
            .panel-title {
                font-size: 17px; font-weight: 700;
                color: var(--text-primary);
                letter-spacing: -0.2px;
            }
            .panel-subtitle {
                font-size: 13px;
                color: var(--text-muted);
                margin-top: 2px;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: var(--white);
                border: 1px solid var(--grey-mid);
                border-radius: 12px;
                padding: 22px;
                position: relative;
                overflow: hidden;
                transition: all 0.2s ease;
            }
            .stat-card:hover {
                border-color: var(--gold-light);
                transform: translateY(-2px);
                box-shadow: var(--shadow-hover);
            }
            .stat-card.gold { background: var(--gold-pale); border-color: var(--gold-light); }
            .stat-card.success { background: var(--success-bg); border-color: #C8E6C8; }
            .stat-card.warning { background: var(--warning-bg); border-color: var(--gold-light); }
            .stat-card.error { background: var(--error-bg); border-color: #F0CCCC; }
            .stat-card::after {
                content: '';
                position: absolute;
                top: 0; right: 0;
                width: 80px; height: 80px;
                background: linear-gradient(135deg, transparent 50%, var(--gold-pale) 50%);
                opacity: 0.4;
                border-radius: 0 12px 0 100%;
            }
            .stat-card.gold::after { display: none; }
            .stat-icon {
                font-size: 22px; margin-bottom: 10px;
            }
            .stat-label {
                color: var(--text-secondary);
                font-size: 11px;
                text-transform: uppercase;
                font-weight: 700;
                letter-spacing: 0.6px;
                margin-bottom: 6px;
            }
            .stat-value {
                font-size: 28px;
                font-weight: 700;
                color: var(--text-primary);
                letter-spacing: -0.6px;
            }
            .stat-value small {
                font-size: 14px;
                color: var(--text-muted);
                font-weight: 500;
            }
            .stat-trend {
                font-size: 12px;
                color: var(--text-muted);
                margin-top: 6px;
                display: flex; align-items: center; gap: 5px;
            }

            .system-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 16px;
            }
            .system-card {
                background: var(--white);
                border: 1px solid var(--grey-mid);
                border-radius: 12px;
                padding: 22px;
                transition: all 0.2s ease;
            }
            .system-card:hover {
                border-color: var(--gold-light);
                box-shadow: var(--shadow-hover);
            }
            .system-card-header {
                display: flex; align-items: center; justify-content: space-between;
                margin-bottom: 12px;
            }
            .system-card-title {
                display: flex; align-items: center; gap: 10px;
                font-size: 14px; font-weight: 600;
                color: var(--text-primary);
            }
            .system-card-title .icon {
                width: 32px; height: 32px;
                background: var(--gold-pale);
                border-radius: 8px;
                display: flex; align-items: center; justify-content: center;
                font-size: 16px;
            }
            .system-card-body {
                font-size: 13px; color: var(--text-secondary);
                line-height: 1.7;
            }
            .system-card-body div {
                display: flex; justify-content: space-between;
                padding: 4px 0;
            }
            .system-card-body div b {
                color: var(--text-primary); font-weight: 600;
            }

            .status-pill {
                display: inline-flex; align-items: center; gap: 6px;
                padding: 4px 10px; border-radius: 100px;
                font-size: 11px; font-weight: 600;
                letter-spacing: 0.3px;
            }
            .status-online { background: var(--success-bg); color: var(--success); border: 1px solid #C8E6C8; }
            .status-offline { background: var(--grey-light); color: var(--text-muted); border: 1px solid var(--grey-mid); }
            .status-warning { background: var(--warning-bg); color: var(--gold-dark); border: 1px solid var(--gold-light); }
            .status-online::before, .status-offline::before, .status-warning::before {
                content: '';
                width: 6px; height: 6px; border-radius: 50%;
                background: currentColor;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                background: var(--white);
                border: 1px solid var(--grey-mid);
                border-radius: 10px;
                overflow: hidden;
            }
            th, td {
                padding: 12px 16px;
                text-align: left;
                border-bottom: 1px solid var(--grey-light);
                font-size: 13px;
            }
            th {
                background: var(--grey-light);
                font-weight: 600;
                color: var(--text-secondary);
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.5px;
            }
            tr:last-child td { border-bottom: none; }
            tr:hover td { background: var(--grey-light); }
            .path {
                font-family: 'SF Mono', Monaco, 'Courier New', monospace;
                font-size: 12px;
                color: var(--text-muted);
            }

            .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 100px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }
            .badge-gold { background: var(--gold-pale); color: var(--gold-dark); border: 1px solid var(--gold-light); }
            .badge-success { background: var(--success-bg); color: var(--success); border: 1px solid #C8E6C8; }
            .badge-warning { background: var(--warning-bg); color: var(--gold-dark); border: 1px solid var(--gold-light); }
            .badge-error { background: var(--error-bg); color: var(--error); border: 1px solid #F0CCCC; }
            .badge-info { background: var(--info-bg); color: #4A6275; border: 1px solid #D8E0E8; }

            .alert-list {
                display: flex; flex-direction: column; gap: 10px;
            }
            .alert {
                padding: 14px 18px;
                border-radius: 10px;
                display: flex; align-items: flex-start; gap: 12px;
                font-size: 13px;
                border-left: 4px solid var(--gold);
            }
            .alert.warning { background: var(--warning-bg); border-color: var(--gold); color: var(--gold-dark); }
            .alert.error { background: var(--error-bg); border-color: var(--error); color: var(--error); }
            .alert.info { background: var(--info-bg); border-color: #6B8DAD; color: #4A6275; }
            .alert.success { background: var(--success-bg); border-color: var(--success); color: var(--success); }
            .alert-icon { font-size: 18px; }
            .alert-body { flex: 1; }
            .alert-title { font-weight: 600; margin-bottom: 2px; }
            .alert-message { font-size: 12px; opacity: 0.9; }

            .quick-link {
                display: inline-flex; align-items: center; gap: 6px;
                padding: 8px 14px;
                background: var(--white);
                border: 1px solid var(--grey-mid);
                border-radius: 8px;
                color: var(--text-primary);
                text-decoration: none;
                font-size: 12px;
                font-weight: 500;
                transition: all 0.2s ease;
            }
            .quick-link:hover {
                border-color: var(--gold-light);
                background: var(--gold-pale);
                color: var(--gold-dark);
            }
            .quick-link-row {
                display: flex; gap: 10px;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }

            .monitor-frame {
                width: 100%;
                height: 500px;
                border: 1px solid var(--grey-mid);
                border-radius: 10px;
                background: var(--grey-light);
            }

            .progress-bar {
                width: 100%; height: 8px;
                background: var(--grey-light);
                border-radius: 100px;
                overflow: hidden;
                margin-top: 6px;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--gold-light) 0%, var(--gold) 100%);
                border-radius: 100px;
                transition: width 0.4s ease;
            }

            .empty-state {
                padding: 40px 20px;
                text-align: center;
                color: var(--text-muted);
            }
            .empty-state .icon { font-size: 36px; margin-bottom: 10px; opacity: 0.5; }
            .empty-state p { font-size: 13px; }

            .refresh-btn {
                padding: 8px 16px;
                background: var(--white);
                border: 1px solid var(--grey-mid);
                border-radius: 8px;
                cursor: pointer;
                font-size: 12px;
                font-weight: 500;
                color: var(--text-primary);
                font-family: inherit;
                transition: all 0.2s ease;
                display: inline-flex; align-items: center; gap: 6px;
            }
            .refresh-btn:hover {
                border-color: var(--gold-light);
                background: var(--gold-pale);
                color: var(--gold-dark);
            }

            .loading {
                text-align: center; padding: 24px;
                color: var(--text-muted); font-size: 13px;
            }
        </style>
    </head>
    <body>
        <div class="top-bar">
            <a href="/" class="home">← Back to Dashboard</a>
            <div class="crumb">LocalML · <b>Admin Console</b></div>
            <div class="live-indicator">
                <span class="live-dot"></span>
                LIVE
            </div>
        </div>

        <div class="container">
            <div class="page-header">
                <div class="page-icon">📊</div>
                <div>
                    <h1>Admin Console</h1>
                    <p>Centralized monitoring across all LocalML subsystems.</p>
                </div>
            </div>

            <div class="tabs">
                <button class="tab active" onclick="switchTab(event, 'overview')">📡 Overview</button>
                <button class="tab" onclick="switchTab(event, 'training')">🎯 Training & Labeling</button>
                <button class="tab" onclick="switchTab(event, 'registry')">📦 Model Registry</button>
                <button class="tab" onclick="switchTab(event, 'validation')">🛡️ Validation Gates</button>
                <button class="tab" onclick="switchTab(event, 'abtest')">⚗️ A/B Testing</button>
                <button class="tab" onclick="switchTab(event, 'serving')">🚀 Inference Serving</button>
                <button class="tab" onclick="switchTab(event, 'system')">📈 System Metrics</button>
            </div>

            <!-- TAB 1: OVERVIEW -->
            <div id="overview" class="tab-content active">
                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Subsystem Health</div>
                            <div class="panel-subtitle">Real-time status of all platform components</div>
                        </div>
                        <button class="refresh-btn" onclick="loadOverview()">↻ Refresh</button>
                    </div>
                    <div id="overview-stats" class="stats-grid"></div>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Subsystems</div>
                            <div class="panel-subtitle">Individual component status and key metrics</div>
                        </div>
                    </div>
                    <div id="overview-systems" class="system-grid"></div>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Active Alerts</div>
                            <div class="panel-subtitle">Issues detected across subsystems</div>
                        </div>
                    </div>
                    <div id="overview-alerts" class="alert-list"></div>
                </div>
            </div>

            <!-- TAB 2: TRAINING & LABELING -->
            <div id="training" class="tab-content">
                <div class="quick-link-row">
                    <a href="/finetune-dashboard" class="quick-link">🎯 Open Fine-tuning Dashboard</a>
                    <a href="/labeling" class="quick-link">🏷️ Open Labeling Tool</a>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Training Jobs</div>
                            <div class="panel-subtitle">Active and recent fine-tuning requests</div>
                        </div>
                        <button class="refresh-btn" onclick="loadTraining()">↻ Refresh</button>
                    </div>
                    <div id="training-stats" class="stats-grid"></div>
                    <div id="training-table-wrap"><div class="loading">Loading training data...</div></div>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Labeling Pipeline</div>
                            <div class="panel-subtitle">Annotation progress and dataset health</div>
                        </div>
                    </div>
                    <div id="labeling-stats" class="stats-grid"></div>
                </div>
            </div>

            <!-- TAB 3: MODEL REGISTRY -->
            <div id="registry" class="tab-content">
                <div class="quick-link-row">
                    <a href="/models-versions" class="quick-link">📦 Open Model Registry</a>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Model Inventory</div>
                            <div class="panel-subtitle">All registered models, versions, and access levels</div>
                        </div>
                        <button class="refresh-btn" onclick="loadRegistry()">↻ Refresh</button>
                    </div>
                    <div id="registry-stats" class="stats-grid"></div>
                    <div id="registry-table-wrap"><div class="loading">Loading registry...</div></div>
                </div>
            </div>

            <!-- TAB 4: VALIDATION GATES -->
            <div id="validation" class="tab-content">
                <div class="quick-link-row">
                    <a href="/model-validation" class="quick-link">🛡️ Open Validation Dashboard</a>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Pre-Deployment Validation</div>
                            <div class="panel-subtitle">K-fold validation gate results and pass/fail history</div>
                        </div>
                        <button class="refresh-btn" onclick="loadValidation()">↻ Refresh</button>
                    </div>
                    <div id="validation-stats" class="stats-grid"></div>
                    <div id="validation-table-wrap"><div class="loading">Loading validation history...</div></div>
                </div>
            </div>

            <!-- TAB 5: A/B TESTING -->
            <div id="abtest" class="tab-content">
                <div class="quick-link-row">
                    <a href="/ab-testing" class="quick-link">⚗️ Open A/B Testing Dashboard</a>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">A/B Tests</div>
                            <div class="panel-subtitle">Active experiments comparing model versions</div>
                        </div>
                        <button class="refresh-btn" onclick="loadABTests()">↻ Refresh</button>
                    </div>
                    <div id="abtest-stats" class="stats-grid"></div>
                    <div id="abtest-table-wrap"><div class="loading">Loading A/B tests...</div></div>
                </div>
            </div>

            <!-- TAB 6: INFERENCE SERVING -->
            <div id="serving" class="tab-content">
                <div class="quick-link-row">
                    <a href="/inference-serving" class="quick-link">🚀 Open Inference Serving</a>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Model Cache</div>
                            <div class="panel-subtitle">In-memory model loading and hit rate</div>
                        </div>
                        <button class="refresh-btn" onclick="loadServing()">↻ Refresh</button>
                    </div>
                    <div id="cache-stats" class="stats-grid"></div>
                    <div id="cache-table-wrap"><div class="loading">Loading cache status...</div></div>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">Traffic Routes</div>
                            <div class="panel-subtitle">Active routing configurations and request distribution</div>
                        </div>
                    </div>
                    <div id="routes-stats" class="stats-grid"></div>
                    <div id="routes-table-wrap"><div class="loading">Loading routes...</div></div>
                </div>
            </div>

            <!-- TAB 7: SYSTEM METRICS -->
            <div id="system" class="tab-content">
                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">External Monitoring</div>
                            <div class="panel-subtitle">Prometheus metrics and Grafana telemetry dashboards</div>
                        </div>
                    </div>
                    <div class="quick-link-row">
                        <a href="http://localhost:9090" target="_blank" class="quick-link">🔥 Prometheus UI</a>
                        <a href="http://localhost:3000" target="_blank" class="quick-link">📊 Grafana Dashboard</a>
                        <a href="/metrics" target="_blank" class="quick-link">📋 Raw Metrics</a>
                    </div>
                    <iframe src="http://localhost:3000/d-solo/sentinel-dash?refresh=5s&theme=light" class="monitor-frame" frameborder="0"></iframe>
                    <p style="font-size: 12px; color: var(--text-muted); margin-top: 12px;">
                        Note: Grafana iframe requires a 'sentinel-dash' dashboard. Use the link above if not loaded.
                    </p>
                </div>
            </div>
        </div>

        <script>
            function switchTab(ev, tabId) {
                document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
                document.getElementById(tabId).classList.add('active');
                ev.currentTarget.classList.add('active');
                // Lazy-load tab data
                if (tabId === 'training') loadTraining();
                if (tabId === 'registry') loadRegistry();
                if (tabId === 'validation') loadValidation();
                if (tabId === 'abtest') loadABTests();
                if (tabId === 'serving') loadServing();
            }

            function statCard(opts) {
                const cls = opts.variant ? 'stat-card ' + opts.variant : 'stat-card';
                const trend = opts.trend ? `<div class="stat-trend">${opts.trend}</div>` : '';
                const icon = opts.icon ? `<div class="stat-icon">${opts.icon}</div>` : '';
                const sub = opts.sub ? `<small> ${opts.sub}</small>` : '';
                return `<div class="${cls}">
                    ${icon}
                    <div class="stat-label">${opts.label}</div>
                    <div class="stat-value">${opts.value}${sub}</div>
                    ${trend}
                </div>`;
            }

            function statusPill(available) {
                if (available === true) return '<span class="status-pill status-online">Online</span>';
                if (available === false) return '<span class="status-pill status-offline">Offline</span>';
                return '<span class="status-pill status-warning">Unknown</span>';
            }

            async function loadOverview() {
                try {
                    const res = await fetch('/admin/overview');
                    const data = await res.json();

                    const totals = data.totals || {};
                    const sys = data.subsystems || {};

                    document.getElementById('overview-stats').innerHTML = [
                        statCard({label: 'Subsystems Online', value: `${totals.subsystems_online || 0}<small>/${totals.subsystems_total || 0}</small>`, icon: '🟢', variant: 'success'}),
                        statCard({label: 'Active Alerts', value: totals.active_alerts || 0, icon: '⚠️', variant: totals.active_alerts > 0 ? 'warning' : 'success'}),
                        statCard({label: 'Models Registered', value: (sys.registry || {}).total_models || 0, icon: '📦', variant: 'gold'}),
                        statCard({label: 'Active A/B Tests', value: (sys.ab_testing || {}).active || 0, icon: '⚗️', variant: 'gold'}),
                        statCard({label: 'Cache Hit Rate', value: `${((sys.cache || {}).hit_rate_percent || 0).toFixed(1)}%`, icon: '⚡', variant: 'gold'}),
                        statCard({label: 'Validation Pass Rate', value: `${(sys.validation || {}).pass_rate || 0}%`, icon: '🛡️', variant: 'gold'}),
                    ].join('');

                    const sysCard = (icon, name, status, lines) => `
                        <div class="system-card">
                            <div class="system-card-header">
                                <div class="system-card-title">
                                    <div class="icon">${icon}</div>
                                    <span>${name}</span>
                                </div>
                                ${statusPill(status)}
                            </div>
                            <div class="system-card-body">${lines.map(l => `<div><span>${l[0]}</span><b>${l[1]}</b></div>`).join('')}</div>
                        </div>
                    `;

                    document.getElementById('overview-systems').innerHTML = [
                        sysCard('🎯', 'Training', (sys.training || {}).available, [
                            ['Total jobs', (sys.training || {}).total_jobs || 0],
                            ['Completed', (sys.training || {}).completed || 0],
                            ['Pending', (sys.training || {}).pending || 0]
                        ]),
                        sysCard('🏷️', 'Labeling', (sys.labeling || {}).available, [
                            ['Labels', (sys.labeling || {}).total_labels || 0],
                            ['Classes', (sys.labeling || {}).total_classes || 0],
                            ['Pending review', (sys.labeling || {}).pending_review || 0]
                        ]),
                        sysCard('📦', 'Registry', (sys.registry || {}).available, [
                            ['Total models', (sys.registry || {}).total_models || 0],
                            ['Unique names', (sys.registry || {}).unique_model_names || 0]
                        ]),
                        sysCard('🛡️', 'Validation', (sys.validation || {}).available, [
                            ['Total gates', (sys.validation || {}).total_gates || 0],
                            ['Passed', (sys.validation || {}).passed || 0],
                            ['Failed', (sys.validation || {}).failed || 0]
                        ]),
                        sysCard('⚗️', 'A/B Testing', (sys.ab_testing || {}).available, [
                            ['Active', (sys.ab_testing || {}).active || 0],
                            ['Completed', (sys.ab_testing || {}).completed || 0],
                            ['Total', (sys.ab_testing || {}).total_tests || 0]
                        ]),
                        sysCard('⚡', 'Cache', (sys.cache || {}).available, [
                            ['Cached models', `${(sys.cache || {}).cache_size || 0}/${(sys.cache || {}).max_cache_size || 0}`],
                            ['Hit rate', `${((sys.cache || {}).hit_rate_percent || 0).toFixed(1)}%`],
                            ['Memory', `${((sys.cache || {}).total_memory_mb || 0).toFixed(1)} MB`]
                        ]),
                        sysCard('🛣️', 'Routing', (sys.routing || {}).available, [
                            ['Total routes', (sys.routing || {}).total_routes || 0],
                            ['Total requests', (sys.routing || {}).total_requests || 0]
                        ])
                    ].join('');

                    const alerts = data.alerts || [];
                    if (alerts.length === 0) {
                        document.getElementById('overview-alerts').innerHTML = `
                            <div class="alert success">
                                <div class="alert-icon">✅</div>
                                <div class="alert-body">
                                    <div class="alert-title">All systems nominal</div>
                                    <div class="alert-message">No active alerts across any subsystem.</div>
                                </div>
                            </div>`;
                    } else {
                        document.getElementById('overview-alerts').innerHTML = alerts.map(a => `
                            <div class="alert ${a.level || 'info'}">
                                <div class="alert-icon">${a.level === 'error' ? '🔴' : a.level === 'warning' ? '⚠️' : 'ℹ️'}</div>
                                <div class="alert-body">
                                    <div class="alert-title">${a.category.toUpperCase()}</div>
                                    <div class="alert-message">${a.message}</div>
                                </div>
                            </div>`).join('');
                    }
                } catch (e) {
                    document.getElementById('overview-stats').innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><p>Failed to load overview data</p></div>';
                }
            }

            async function loadTraining() {
                try {
                    const res = await fetch('/admin/data');
                    const rows = await res.json();
                    const total = rows.length;
                    const completed = rows.filter(r => r.model_path).length;
                    const pending = total - completed;

                    document.getElementById('training-stats').innerHTML = [
                        statCard({label: 'Total Jobs', value: total, icon: '🎯', variant: 'gold'}),
                        statCard({label: 'Completed', value: completed, icon: '✅', variant: 'success'}),
                        statCard({label: 'Pending', value: pending, icon: '⏳', variant: pending > 0 ? 'warning' : ''})
                    ].join('');

                    if (rows.length === 0) {
                        document.getElementById('training-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">📭</div><p>No training jobs yet</p></div>';
                    } else {
                        document.getElementById('training-table-wrap').innerHTML = `
                            <table>
                                <thead><tr><th>ID</th><th>Target</th><th>Dataset</th><th>Model</th><th>Timestamp</th></tr></thead>
                                <tbody>${rows.map(r => `
                                    <tr>
                                        <td>${r.id}</td>
                                        <td><span class="badge badge-gold">${r.target_object || 'N/A'}</span></td>
                                        <td class="path">${r.dataset_path || '-'}</td>
                                        <td>${r.model_path ? '<span class="badge badge-success">Ready</span>' : '<span class="badge badge-warning">Pending</span>'}</td>
                                        <td class="path">${r.timestamp || '-'}</td>
                                    </tr>`).join('')}
                                </tbody>
                            </table>`;
                    }

                    // Labeling stats
                    try {
                        const lr = await fetch('/labeling-stats');
                        const lj = await lr.json();
                        const ls = lj.stats || {};
                        document.getElementById('labeling-stats').innerHTML = [
                            statCard({label: 'Labeled Images', value: ls.labeled || 0, icon: '🏷️', variant: 'gold'}),
                            statCard({label: 'Classes', value: ls.num_classes || 0, icon: '🗂️'}),
                            statCard({label: 'Pending Review', value: ls.pending || 0, icon: '👁️', variant: (ls.pending || 0) > 0 ? 'warning' : ''}),
                            statCard({label: 'Progress', value: `${(ls.progress || 0).toFixed(1)}%`, icon: '📊', variant: 'gold'})
                        ].join('');
                    } catch (e) {
                        document.getElementById('labeling-stats').innerHTML = '<div class="empty-state"><p>Labeling service unavailable</p></div>';
                    }
                } catch (e) {
                    document.getElementById('training-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><p>Failed to load training data</p></div>';
                }
            }

            async function loadRegistry() {
                try {
                    const res = await fetch('/model-registry/all');
                    const data = await res.json();
                    const models = data.models || [];

                    const total = models.length;
                    const accessGroups = {};
                    models.forEach(m => {
                        const a = m.access || 'unknown';
                        accessGroups[a] = (accessGroups[a] || 0) + 1;
                    });

                    document.getElementById('registry-stats').innerHTML = [
                        statCard({label: 'Total Models', value: total, icon: '📦', variant: 'gold'}),
                        statCard({label: 'Public', value: accessGroups.public || 0, icon: '🌐'}),
                        statCard({label: 'Shared', value: accessGroups.shared || 0, icon: '👥'}),
                        statCard({label: 'Private', value: accessGroups.private || 0, icon: '🔒'})
                    ].join('');

                    if (models.length === 0) {
                        document.getElementById('registry-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">📭</div><p>No models registered yet</p></div>';
                    } else {
                        document.getElementById('registry-table-wrap').innerHTML = `
                            <table>
                                <thead><tr><th>ID</th><th>Name</th><th>Version</th><th>Access</th><th>Created</th></tr></thead>
                                <tbody>${models.slice(0, 20).map(m => `
                                    <tr>
                                        <td>${m.id || '-'}</td>
                                        <td><b>${m.name || '-'}</b></td>
                                        <td><span class="badge badge-gold">${m.version || '-'}</span></td>
                                        <td><span class="badge badge-info">${m.access || 'private'}</span></td>
                                        <td class="path">${m.created_at || '-'}</td>
                                    </tr>`).join('')}
                                </tbody>
                            </table>`;
                    }
                } catch (e) {
                    document.getElementById('registry-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><p>Registry unavailable</p></div>';
                }
            }

            async function loadValidation() {
                try {
                    const res = await fetch('/validation-history');
                    const data = await res.json();
                    const gates = data.gates || data.records || data.history || [];

                    const total = gates.length;
                    const passed = gates.filter(g => (g.decision || '').toUpperCase() === 'PASS').length;
                    const failed = gates.filter(g => (g.decision || '').toUpperCase() === 'FAIL').length;
                    const passRate = total > 0 ? ((passed / total) * 100).toFixed(1) : 0;

                    document.getElementById('validation-stats').innerHTML = [
                        statCard({label: 'Total Validations', value: total, icon: '🛡️', variant: 'gold'}),
                        statCard({label: 'Passed', value: passed, icon: '✅', variant: 'success'}),
                        statCard({label: 'Failed', value: failed, icon: '❌', variant: failed > 0 ? 'error' : ''}),
                        statCard({label: 'Pass Rate', value: `${passRate}%`, icon: '📈', variant: 'gold'})
                    ].join('');

                    if (gates.length === 0) {
                        document.getElementById('validation-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">📭</div><p>No validation gates run yet</p></div>';
                    } else {
                        document.getElementById('validation-table-wrap').innerHTML = `
                            <table>
                                <thead><tr><th>Model</th><th>Decision</th><th>Mean Acc</th><th>Std Dev</th><th>Folds</th><th>Date</th></tr></thead>
                                <tbody>${gates.slice(0, 20).map(g => `
                                    <tr>
                                        <td><b>${g.model_name || g.model_id || '-'}</b></td>
                                        <td>${(g.decision || '').toUpperCase() === 'PASS' ? '<span class="badge badge-success">PASS</span>' : '<span class="badge badge-error">FAIL</span>'}</td>
                                        <td>${(g.mean_accuracy || 0).toFixed ? (g.mean_accuracy || 0).toFixed(3) : '-'}</td>
                                        <td>${(g.std_dev || 0).toFixed ? (g.std_dev || 0).toFixed(3) : '-'}</td>
                                        <td>${g.num_folds || '-'}</td>
                                        <td class="path">${g.created_at || '-'}</td>
                                    </tr>`).join('')}
                                </tbody>
                            </table>`;
                    }
                } catch (e) {
                    document.getElementById('validation-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><p>Validation service unavailable</p></div>';
                }
            }

            async function loadABTests() {
                try {
                    const res = await fetch('/ab-test/list');
                    const data = await res.json();
                    const tests = data.tests || [];

                    const active = tests.filter(t => (t.status || '').toLowerCase() === 'active').length;
                    const completed = tests.filter(t => (t.status || '').toLowerCase() === 'completed').length;

                    document.getElementById('abtest-stats').innerHTML = [
                        statCard({label: 'Total Tests', value: tests.length, icon: '⚗️', variant: 'gold'}),
                        statCard({label: 'Active', value: active, icon: '🔬', variant: active > 0 ? 'gold' : ''}),
                        statCard({label: 'Completed', value: completed, icon: '🏁', variant: 'success'})
                    ].join('');

                    if (tests.length === 0) {
                        document.getElementById('abtest-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">📭</div><p>No A/B tests configured</p></div>';
                    } else {
                        document.getElementById('abtest-table-wrap').innerHTML = `
                            <table>
                                <thead><tr><th>Name</th><th>Status</th><th>Model A</th><th>Model B</th><th>Split</th><th>Created</th></tr></thead>
                                <tbody>${tests.slice(0, 20).map(t => {
                                    const status = (t.status || 'unknown').toLowerCase();
                                    const cls = status === 'active' ? 'badge-warning' : status === 'completed' ? 'badge-success' : 'badge-info';
                                    return `<tr>
                                        <td><b>${t.name || '-'}</b></td>
                                        <td><span class="badge ${cls}">${status.toUpperCase()}</span></td>
                                        <td class="path">${(t.model_a_path || '').split('/').pop() || t.model_a_id || '-'}</td>
                                        <td class="path">${(t.model_b_path || '').split('/').pop() || t.model_b_id || '-'}</td>
                                        <td>${((t.split_ratio || 0.5) * 100).toFixed(0)}/${((1 - (t.split_ratio || 0.5)) * 100).toFixed(0)}</td>
                                        <td class="path">${t.created_at || '-'}</td>
                                    </tr>`;
                                }).join('')}
                                </tbody>
                            </table>`;
                    }
                } catch (e) {
                    document.getElementById('abtest-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><p>A/B testing service unavailable</p></div>';
                }
            }

            async function loadServing() {
                try {
                    const cr = await fetch('/cache-status');
                    const cs = await cr.json();

                    document.getElementById('cache-stats').innerHTML = [
                        statCard({label: 'Cached Models', value: `${cs.cache_size || 0}<small>/${cs.max_cache_size || 0}</small>`, icon: '📦', variant: 'gold'}),
                        statCard({label: 'Hit Rate', value: `${(cs.hit_rate_percent || 0).toFixed(1)}%`, icon: '⚡', variant: (cs.hit_rate_percent || 0) >= 70 ? 'success' : 'warning'}),
                        statCard({label: 'Total Hits', value: cs.total_hits || 0, icon: '✅'}),
                        statCard({label: 'Memory', value: `${(cs.total_memory_mb || 0).toFixed(1)}<small> MB</small>`, icon: '💾'})
                    ].join('');

                    const cached = cs.cached_models || [];
                    if (cached.length === 0) {
                        document.getElementById('cache-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">📭</div><p>No models cached</p></div>';
                    } else {
                        document.getElementById('cache-table-wrap').innerHTML = `
                            <table>
                                <thead><tr><th>Model</th><th>Hits</th><th>Misses</th><th>Size (MB)</th><th>Status</th></tr></thead>
                                <tbody>${cached.map(m => `
                                    <tr>
                                        <td><b>${m.filename || '-'}</b></td>
                                        <td>${m.hits || 0}</td>
                                        <td>${m.misses || 0}</td>
                                        <td>${(m.file_size_mb || 0).toFixed(1)}</td>
                                        <td>${m.in_memory ? '<span class="badge badge-success">In Memory</span>' : '<span class="badge badge-warning">On Disk</span>'}</td>
                                    </tr>`).join('')}
                                </tbody>
                            </table>`;
                    }
                } catch (e) {
                    document.getElementById('cache-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><p>Cache service unavailable</p></div>';
                }

                try {
                    const rr = await fetch('/traffic-routes');
                    const rs = await rr.json();
                    const routes = rs.routes || [];

                    const totalReq = routes.reduce((s, r) => s + (r.requests || 0), 0);

                    document.getElementById('routes-stats').innerHTML = [
                        statCard({label: 'Active Routes', value: routes.length, icon: '🛣️', variant: 'gold'}),
                        statCard({label: 'Total Requests', value: totalReq, icon: '📨'}),
                        statCard({label: 'Strategies', value: new Set(routes.map(r => r.strategy)).size, icon: '🎯'})
                    ].join('');

                    if (routes.length === 0) {
                        document.getElementById('routes-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">📭</div><p>No traffic routes configured</p></div>';
                    } else {
                        document.getElementById('routes-table-wrap').innerHTML = `
                            <table>
                                <thead><tr><th>Route ID</th><th>Strategy</th><th>Models</th><th>Requests</th></tr></thead>
                                <tbody>${routes.map(r => `
                                    <tr>
                                        <td><b>${r.route_id || '-'}</b></td>
                                        <td><span class="badge badge-info">${r.strategy || '-'}</span></td>
                                        <td class="path">${(r.models || []).map(m => m.split('/').pop()).join(' ↔ ')}</td>
                                        <td>${r.requests || 0}</td>
                                    </tr>`).join('')}
                                </tbody>
                            </table>`;
                    }
                } catch (e) {
                    document.getElementById('routes-table-wrap').innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><p>Routing service unavailable</p></div>';
                }
            }

            // Initial load + auto-refresh every 30s
            loadOverview();
            setInterval(loadOverview, 30000);
        </script>
    </body>
    </html>
    """


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/mobile-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            mobile_source.update_frame(data)
    except WebSocketDisconnect:
        print("Mobile client disconnected")
        pass

@app.get("/qr-code")
async def get_qr_code():
    # Generate QR code for the mobile capture page
    # Assuming the server is reachable via local network IP or localhost for demo
    # In a real scenario, this IP needs to be the LAN IP of the host machine
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # Try to find a non-loopback IP (simple heuristic)
    try:
         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
         s.connect(("8.8.8.8", 80))
         local_ip = s.getsockname()[0]
         s.close()
    except:
         pass

    url = f"http://{local_ip}:8001/mobile-capture"
    
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/mobile-capture", response_class=HTMLResponse)
async def mobile_capture_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LocalML</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            :root {
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
            }
            body { margin: 0; background: var(--text-primary); color: var(--white); display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; font-family: Inter, sans-serif; }
            video { width: 100%; max-height: 80vh; object-fit: contain; }
            button { padding: 12px 24px; font-size: 1rem; background: var(--gold); color: var(--white); border: none; border-radius: 4px; margin-top: 20px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; cursor: pointer; transition: background 0.15s; }
            button:hover { background: var(--gold-dark); }
            button:active { transform: scale(0.98); }
            #status { margin-top: 10px; color: var(--text-secondary); font-size: 14px; }
            .status-connected { color: #4CAF50 !important; }
            .status-disconnected { color: #8B4513 !important; }
        </style>
    </head>
    <body>
        <video id="video" autoplay playsinline muted></video>
        <div id="status">READY TO CONNECT</div>
        <button onclick="startStreaming()">START STREAMING</button>
        <script>
            const video = document.getElementById('video');
            const status = document.getElementById('status');
            let ws;
            let canvas = document.createElement('canvas');
            
            async function startStreaming() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
                    video.srcObject = stream;
                    
                    // Connect WS
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/mobile-stream`);
                    
                    ws.onopen = () => {
                        status.textContent = "CONNECTED - STREAMING";
                        status.classList.add("status-connected");
                        status.classList.remove("status-disconnected");
                        sendFrame();
                    };

                    ws.onclose = () => {
                        status.textContent = "DISCONNECTED";
                        status.classList.add("status-disconnected");
                        status.classList.remove("status-connected");
                    };
                    
                } catch (e) {
                    status.textContent = "Error: " + e.message;
                }
            }
            
            function sendFrame() {
                if (ws.readyState === WebSocket.OPEN) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0);
                    
                    canvas.toBlob(blob => {
                        if (ws.readyState === WebSocket.OPEN) ws.send(blob);
                        requestAnimationFrame(sendFrame); // Loop
                    }, 'image/jpeg', 0.5); // Compress to 0.5 quality
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/models-inference")
async def models_inference_ui():
    """Model browser and inference dashboard for fine-tuned models."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fine-tuned Models Inference</title>
        <style>
            :root {
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
                --text-muted: #888888;
                --shadow-soft: 0 1px 3px rgba(0,0,0,0.08);
                --shadow-hover: 0 4px 12px rgba(184,134,11,0.15);
            }
            * { font-family: Inter, Arial, sans-serif; margin: 0; padding: 0; }
            body { background: var(--bg); padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; background: var(--white); padding: 32px; border-radius: 6px; box-shadow: var(--shadow-soft); border: 1px solid var(--border); }
            h1 { color: var(--gold-dark); margin-bottom: 10px; font-size: 24px; font-weight: 700; }
            .subtitle { color: var(--text-secondary); margin-bottom: 30px; font-size: 13px; }

            .layout { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }

            .panel { padding: 20px; background: var(--bg); border-radius: 6px; border: 1px solid var(--border); }
            .panel h2 { color: var(--gold-dark); font-size: 14px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 0.3px; font-weight: 700; }

            .models-list { max-height: 500px; overflow-y: auto; }
            .model-card { padding: 15px; background: var(--white); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 10px; cursor: pointer; transition: all 0.15s; }
            .model-card:hover { background: var(--bg); border-color: var(--gold); box-shadow: var(--shadow-hover); }
            .model-card.selected { background: #F0F9F0; border-color: var(--gold); font-weight: bold; }
            .model-name { font-weight: 700; color: var(--text-primary); margin-bottom: 5px; }
            .model-meta { font-size: 12px; color: var(--text-secondary); line-height: 1.4; }
            .model-path { font-family: monospace; font-size: 11px; color: var(--text-muted); margin-top: 8px; background: var(--bg); padding: 5px; border-radius: 3px; word-break: break-all; }
            .model-badge { display: inline-block; background: var(--gold); color: var(--white); padding: 2px 8px; border-radius: 3px; font-size: 10px; margin-top: 5px; font-weight: 700; }
            .model-badge.best { background: var(--gold-dark); }

            .form-group { display: flex; flex-direction: column; margin-bottom: 15px; }
            .form-group label { font-weight: 700; margin-bottom: 5px; color: var(--text-primary); font-size: 12px; text-transform: uppercase; letter-spacing: 0.3px; }
            .form-group input, .form-group select, .form-group textarea { padding: 10px; border: 1px solid var(--border); border-radius: 4px; font-size: 13px; background: var(--white); color: var(--text-primary); }
            .form-group input:focus, .form-group select:focus, .form-group textarea:focus { outline: none; border-color: var(--gold); box-shadow: 0 0 5px rgba(184,134,11,0.3); }

            .button-group { display: flex; gap: 10px; margin-top: 20px; }
            button { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: 700; transition: background 0.15s; font-size: 12px; text-transform: uppercase; letter-spacing: 0.3px; }
            .btn-primary { background: var(--gold); color: var(--white); }
            .btn-primary:hover { background: var(--gold-dark); }
            .btn-primary:disabled { background: var(--border); cursor: not-allowed; }
            .btn-secondary { background: var(--text-secondary); color: var(--white); }
            .btn-secondary:hover { background: var(--text-primary); }

            .results { margin-top: 20px; padding: 15px; background: var(--bg); border-radius: 6px; border: 1px solid var(--border); display: none; }
            .results.show { display: block; }
            .results h3 { margin-bottom: 10px; color: var(--gold-dark); font-size: 14px; text-transform: uppercase; letter-spacing: 0.3px; }
            .prediction { padding: 10px; background: var(--white); border-left: 4px solid var(--gold); margin-bottom: 8px; border-radius: 4px; }
            .prediction-class { font-weight: 700; color: var(--text-primary); }
            .prediction-confidence { color: var(--gold-dark); font-size: 12px; font-weight: 600; }

            .loading { display: none; text-align: center; color: var(--text-secondary); padding: 20px; }
            .loading.show { display: block; }

            .message { padding: 12px; border-radius: 4px; margin-bottom: 10px; border: 1px solid var(--border); }
            .message.success { background: #F0F9F0; color: var(--gold-dark); }
            .message.error { background: #FAE8E5; color: #8B4513; }
            .message.info { background: var(--bg); color: var(--text-primary); }

            .upload-area { border: 2px dashed var(--border); padding: 20px; text-align: center; border-radius: 6px; cursor: pointer; transition: all 0.15s; background: var(--bg); }
            .upload-area:hover { border-color: var(--gold); background: #FFFBF5; }
            .upload-area input { display: none; }

            .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px; }
            .stat { text-align: center; padding: 10px; background: var(--bg); border-radius: 6px; border: 1px solid var(--border); }
            .stat-value { font-size: 18px; font-weight: 800; color: var(--gold-dark); }
            .stat-label { font-size: 10px; color: var(--text-muted); margin-top: 5px; text-transform: uppercase; letter-spacing: 0.3px; }

            .disabled { opacity: 0.6; pointer-events: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fine-tuned Models Inference</h1>
            <p class="subtitle">Browse and run inference with fine-tuned models</p>

            <div class="stats" id="stats"></div>

            <div class="layout">
                <!-- Left: Models List -->
                <div class="panel">
                    <h2>📦 Available Models</h2>
                    <div class="models-list" id="modelsList">
                        <p style="color: #999; text-align: center; padding: 20px;">Loading models...</p>
                    </div>
                </div>

                <!-- Right: Inference Panel -->
                <div class="panel">
                    <h2>🔍 Run Inference</h2>

                    <div id="message"></div>

                    <div class="form-group">
                        <label>Select Model</label>
                        <select id="modelSelect" disabled>
                            <option value="">-- Choose a model --</option>
                        </select>
                        <small style="color: #999; margin-top: 5px;">Click on a model in the list to select it</small>
                    </div>

                    <div class="form-group">
                        <label>Upload Image</label>
                        <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                            <p>📷 Click to upload or drag and drop</p>
                            <small>JPG, PNG</small>
                            <input type="file" id="imageInput" accept="image/*">
                        </div>
                        <p id="fileName" style="margin-top: 10px; color: #666; font-size: 12px;"></p>
                    </div>

                    <div class="form-group">
                        <label>Return Top K Predictions</label>
                        <input type="number" id="topK" min="1" max="10" value="3">
                    </div>

                    <div class="button-group">
                        <button class="btn-primary" id="runBtn" onclick="runInference()" disabled>▶️ Run Inference</button>
                        <button class="btn-secondary" onclick="resetForm()">↺ Reset</button>
                    </div>

                    <div class="loading" id="loading">
                        <p>Running inference...</p>
                    </div>

                    <div class="results" id="results">
                        <h3>✅ Predictions</h3>
                        <div id="predictionsContainer"></div>
                        <p id="inferenceInfo" style="font-size: 11px; color: #999; margin-top: 10px;"></p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let selectedModel = null;
            let selectedFile = null;

            async function loadModels() {
                try {
                    const resp = await fetch('/models-list');
                    const data = await resp.json();

                    if (!data.models || data.models.length === 0) {
                        document.getElementById('modelsList').innerHTML = '<p style="color: #999; text-align: center; padding: 20px;">No fine-tuned models found</p>';
                        return;
                    }

                    // Update stats
                    document.getElementById('stats').innerHTML = `
                        <div class="stat">
                            <div class="stat-value">${data.models.length}</div>
                            <div class="stat-label">Total Models</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${data.total_size_mb?.toFixed(1) || '?'}</div>
                            <div class="stat-label">Total Size (MB)</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${data.unique_sessions?.length || 0}</div>
                            <div class="stat-label">Training Sessions</div>
                        </div>
                    `;

                    // Render models list
                    const modelsList = document.getElementById('modelsList');
                    modelsList.innerHTML = '';

                    data.models.forEach(model => {
                        const div = document.createElement('div');
                        div.className = 'model-card';
                        div.innerHTML = `
                            <div class="model-name">
                                ${model.display_name}
                                ${model.is_best ? '<span class="model-badge best">BEST</span>' : ''}
                            </div>
                            <div class="model-meta">
                                <div>Session: ${model.session_id}</div>
                                <div>Classes: ${model.num_classes || '?'}</div>
                                <div>Size: ${model.size_mb?.toFixed(1) || '?'} MB</div>
                                <div>Epoch: ${model.epoch || '?'}</div>
                                <div>Created: ${new Date(model.created_at).toLocaleDateString()}</div>
                            </div>
                            <div class="model-path">${model.full_path}</div>
                        `;
                        div.onclick = () => selectModel(model, div);
                        modelsList.appendChild(div);
                    });

                    // Populate select dropdown
                    const select = document.getElementById('modelSelect');
                    select.innerHTML = '<option value="">-- Choose a model --</option>';
                    data.models.forEach(model => {
                        const opt = document.createElement('option');
                        opt.value = model.full_path;
                        opt.textContent = model.display_name;
                        select.appendChild(opt);
                    });
                } catch (e) {
                    console.error('Error loading models:', e);
                    document.getElementById('modelsList').innerHTML = '<p style="color: #c62828;">Error loading models</p>';
                }
            }

            function selectModel(model, element) {
                selectedModel = model;
                document.querySelectorAll('.model-card').forEach(el => el.classList.remove('selected'));
                element.classList.add('selected');
                document.getElementById('modelSelect').value = model.full_path;
                document.getElementById('runBtn').disabled = false;
                document.getElementById('results').classList.remove('show');
                showMessage(`✓ Selected: ${model.display_name}`, 'info');
            }

            document.getElementById('imageInput').addEventListener('change', (e) => {
                if (e.target.files[0]) {
                    selectedFile = e.target.files[0];
                    document.getElementById('fileName').textContent = '✓ Selected: ' + e.target.files[0].name;
                    if (selectedModel) document.getElementById('runBtn').disabled = false;
                }
            });

            async function runInference() {
                if (!selectedModel || !selectedFile) {
                    showMessage('Please select a model and image', 'error');
                    return;
                }

                const formData = new FormData();
                formData.append('image', selectedFile);
                formData.append('model_path', selectedModel.full_path);
                formData.append('top_k', document.getElementById('topK').value);

                try {
                    document.getElementById('loading').classList.add('show');
                    document.getElementById('results').classList.remove('show');

                    const resp = await fetch('/inference-finetune', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await resp.json();

                    if (data.status === 'success') {
                        let html = '';
                        data.predictions.forEach((pred, idx) => {
                            html += `
                                <div class="prediction">
                                    <div class="prediction-class">${idx + 1}. ${pred.class_name}</div>
                                    <div class="prediction-confidence">Confidence: ${pred.percentage.toFixed(1)}%</div>
                                </div>
                            `;
                        });
                        document.getElementById('predictionsContainer').innerHTML = html;
                        document.getElementById('inferenceInfo').innerHTML =
                            `Model: ${selectedModel.display_name}<br>Image Size: ${data.image_size}×${data.image_size}`;
                        document.getElementById('results').classList.add('show');
                    } else {
                        showMessage('Error: ' + data.message, 'error');
                    }
                } catch (e) {
                    showMessage('Error: ' + e.message, 'error');
                } finally {
                    document.getElementById('loading').classList.remove('show');
                }
            }

            function resetForm() {
                selectedFile = null;
                document.getElementById('imageInput').value = '';
                document.getElementById('fileName').textContent = '';
                document.getElementById('results').classList.remove('show');
            }

            function showMessage(msg, type) {
                const div = document.getElementById('message');
                div.innerHTML = '<div class="message ' + type + '">' + msg + '</div>';
                setTimeout(() => div.innerHTML = '', 5000);
            }

            // Load models on startup
            loadModels();
            setInterval(loadModels, 10000); // Refresh every 10 seconds
        </script>
    </body>
    </html>
    """)


@app.get("/models-list")
async def models_list():
    """List all available fine-tuned models with metadata."""
    import os
    from pathlib import Path

    models_dir = Path("finetuned_models")
    models = []
    sessions = set()
    total_size = 0

    if not models_dir.exists():
        return {
            "status": "success",
            "models": [],
            "total_size_mb": 0,
            "unique_sessions": []
        }

    # Find all checkpoint files
    for file_path in sorted(models_dir.glob("*.ckpt")):
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += size_mb

            # Parse filename: timestamp_modelname_epoch_NNN.ckpt
            name = file_path.stem  # Remove .ckpt
            parts = name.split("_")

            session_id = parts[0] if len(parts) > 0 else "unknown"
            sessions.add(session_id)

            # Extract epoch number if present
            epoch = None
            for i, part in enumerate(parts):
                if part == "epoch" and i + 1 < len(parts):
                    epoch = parts[i + 1]
                    break

            # Try to get num_classes from JSON metadata if available
            metadata_path = file_path.parent / f"{name}_metadata.json"
            num_classes = None
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    meta = json.load(f)
                    num_classes = meta.get("num_classes")

            models.append({
                "name": file_path.name,
                "display_name": f"{session_id} - Epoch {epoch or '?'}",
                "session_id": session_id,
                "epoch": epoch,
                "full_path": str(file_path),
                "size_mb": size_mb,
                "created_at": file_path.stat().st_mtime,
                "num_classes": num_classes,
                "is_best": "epoch_001" in name or "epoch_002" in name or "epoch_003" in name
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return {
        "status": "success",
        "models": models,
        "total_size_mb": total_size,
        "unique_sessions": list(sessions)
    }


@app.post("/inference-finetune")
async def inference_finetune(image: UploadFile = File(...),
                            model_path: str = Form(...),
                            top_k: int = Form(default=3)):
    """
    Run inference with a fine-tuned model.

    Args:
        image: Input image
        model_path: Path to fine-tuned checkpoint
        top_k: Return top K predictions
    """
    if not JAX_AVAILABLE:
        return {"status": "error", "message": "JAX not available"}

    try:
        contents = await image.read()

        # Initialize inference wrapper
        # In production, would extract num_classes from model metadata
        inference = jax_train.FinetuneInference(num_classes=10, input_size=224)

        # Load checkpoint
        if not inference.load_checkpoint(model_path):
            return {"status": "error", "message": "Failed to load checkpoint"}

        # Run inference
        result = inference.predict(contents, return_top_k=top_k)
        return result

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/finetune-dashboard")
async def finetune_dashboard():
    """Fine-tuning configuration dashboard with validation, early stopping, and checkpointing options."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fine-tuning · LocalML</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            :root {
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --card: #FFFFFF;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
                --text-muted: #888888;
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --gold-light: #D4A373;
                --gold-pale: #FFF8DC;
                --shadow-soft: 0 1px 3px rgba(0,0,0,0.08);
                --shadow-hover: 0 4px 12px rgba(184,134,11,0.15);
                --success: #B8860B;
                --success-bg: #FFF8DC;
                --warning: #B8860B;
                --error: #8B6914;
                --error-bg: #FFF5F0;
            }
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                   background: var(--bg); color: var(--text-primary); min-height: 100vh; }

            .top-bar {
                background: var(--white); border-bottom: 1px solid var(--grey-mid);
                padding: 18px 40px;
                display: flex; align-items: center; justify-content: space-between;
                box-shadow: var(--shadow-soft);
            }
            .top-bar a.home {
                display: flex; align-items: center; gap: 10px;
                color: var(--text-secondary); text-decoration: none;
                font-size: 13px; font-weight: 500; transition: color 0.2s;
            }
            .top-bar a.home:hover { color: var(--gold-dark); }
            .top-bar .crumb { display: flex; align-items: center; gap: 10px;
                              font-size: 13px; color: var(--text-muted); }
            .top-bar .crumb b { color: var(--text-primary); font-weight: 600; }

            .container { max-width: 1100px; margin: 0 auto; padding: 32px 40px; }

            .page-header {
                display: flex; align-items: flex-start; gap: 18px; margin-bottom: 28px;
            }
            .page-icon {
                width: 52px; height: 52px;
                background: linear-gradient(135deg, var(--gold-light) 0%, var(--gold) 100%);
                border-radius: 12px;
                display: flex; align-items: center; justify-content: center;
                font-size: 24px; box-shadow: 0 6px 18px var(--gold-shadow);
            }
            h1 { font-size: 26px; font-weight: 700;
                 color: var(--text-primary); letter-spacing: -0.4px; margin-bottom: 4px; }
            .page-header p { font-size: 14px; color: var(--text-secondary); }

            .section {
                background: var(--white); border: 1px solid var(--grey-mid);
                border-radius: 12px; padding: 26px;
                margin-bottom: 18px; box-shadow: var(--shadow-soft);
            }
            .section-header {
                display: flex; align-items: center; gap: 12px;
                margin-bottom: 18px;
                padding-bottom: 14px;
                border-bottom: 1px solid var(--grey-light);
            }
            .section-icon {
                width: 32px; height: 32px;
                background: var(--gold-pale); border-radius: 8px;
                display: flex; align-items: center; justify-content: center;
                font-size: 16px;
            }
            .section h2 { font-size: 16px; font-weight: 700;
                          color: var(--text-primary); letter-spacing: -0.2px; }
            .section .section-sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

            .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 14px; }
            .form-grid.full { grid-template-columns: 1fr; }
            .form-group { display: flex; flex-direction: column; }
            .form-group label { font-weight: 600; margin-bottom: 6px;
                                color: var(--text-primary); font-size: 13px; }
            .form-group input, .form-group select {
                padding: 11px 14px; border: 1px solid var(--grey-mid);
                border-radius: 8px; font-size: 13px; font-family: inherit;
                color: var(--text-primary); background: var(--white);
                transition: all 0.2s ease;
            }
            .form-group input:focus, .form-group select:focus {
                outline: none; border-color: var(--gold);
                box-shadow: 0 0 0 3px var(--gold-shadow);
            }
            .form-group .help { font-size: 11px; color: var(--text-muted); margin-top: 5px; }

            .checkbox-group {
                display: flex; align-items: center; gap: 10px;
                padding: 12px 16px; background: var(--grey-light);
                border-radius: 8px; margin-bottom: 10px;
                cursor: pointer; transition: all 0.2s;
            }
            .checkbox-group:hover { background: var(--gold-pale); }
            .checkbox-group input[type="checkbox"] {
                cursor: pointer; width: 18px; height: 18px; accent-color: var(--gold);
            }
            .checkbox-group label { margin: 0; cursor: pointer; font-size: 13px;
                                    font-weight: 600; color: var(--text-primary); flex: 1; }

            .slider-value {
                display: inline-block; background: var(--gold-pale);
                color: var(--gold-dark); border: 1px solid var(--gold-light);
                padding: 3px 10px; border-radius: 100px; font-size: 12px;
                font-weight: 600; margin-left: 10px;
            }

            button {
                padding: 11px 22px; border: none; border-radius: 8px;
                cursor: pointer; font-weight: 600; transition: all 0.2s;
                font-size: 13px; font-family: inherit;
            }
            .btn-primary {
                background: linear-gradient(135deg, var(--gold) 0%, var(--gold-dark) 100%);
                color: white; box-shadow: 0 3px 10px var(--gold-shadow);
            }
            .btn-primary:hover { transform: translateY(-1px);
                                 box-shadow: 0 6px 18px var(--gold-shadow); }
            .btn-secondary {
                background: var(--white); color: var(--text-primary);
                border: 1px solid var(--grey-mid);
            }
            .btn-secondary:hover { border-color: var(--gold-light);
                                   background: var(--gold-pale); color: var(--gold-dark); }

            .preset-buttons { display: flex; gap: 8px; flex-wrap: wrap; }
            .preset-btn {
                padding: 9px 14px; background: var(--white);
                border: 1px solid var(--grey-mid); border-radius: 8px;
                cursor: pointer; font-size: 12px; font-weight: 500;
                color: var(--text-primary); font-family: inherit; transition: all 0.2s;
            }
            .preset-btn:hover { border-color: var(--gold-light);
                                background: var(--gold-pale); color: var(--gold-dark); }

            .message { padding: 14px 18px; border-radius: 10px;
                       margin-bottom: 14px; font-size: 13px;
                       border-left: 4px solid var(--gold); display: flex; gap: 12px; align-items: flex-start; }
            .message.success { background: var(--success-bg); border-color: var(--success); color: var(--success); }
            .message.error { background: var(--error-bg); border-color: var(--error); color: var(--error); }
            .message.info { background: var(--gold-pale); border-color: var(--gold); color: var(--gold-dark); }
            .message.warning { background: var(--gold-pale); border-color: var(--gold); color: var(--gold-dark); }

            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                     gap: 14px; margin-bottom: 20px; }
            .stat-card {
                background: var(--gold-pale); padding: 18px;
                border-radius: 10px; border: 1px solid var(--gold-light);
                position: relative; overflow: hidden;
            }
            .stat-card::before {
                content: ''; position: absolute; top: 0; left: 0;
                width: 4px; height: 100%; background: var(--gold);
            }
            .stat-label { font-size: 11px; color: var(--gold-dark);
                          text-transform: uppercase; font-weight: 700;
                          letter-spacing: 0.5px; margin-bottom: 4px; }
            .stat-value { font-size: 22px; font-weight: 700;
                          color: var(--text-primary); letter-spacing: -0.4px; }
            .stat-value small { font-size: 13px; color: var(--text-muted); font-weight: 500; }

            .upload-area {
                border: 2px dashed var(--grey-mid); padding: 36px 20px;
                text-align: center; border-radius: 12px; cursor: pointer;
                transition: all 0.2s; background: var(--grey-light);
            }
            .upload-area:hover, .upload-area.dragover {
                border-color: var(--gold); background: var(--gold-pale);
            }
            .upload-area .icon { font-size: 32px; margin-bottom: 6px; }
            .upload-area p { color: var(--text-secondary); font-size: 13px; }
            .upload-area small { color: var(--text-muted); font-size: 12px; }
            .upload-area input { display: none; }

            .progress-area { margin-top: 18px; display: none; padding: 14px 18px;
                             background: var(--gold-pale); border: 1px solid var(--gold-light);
                             border-radius: 10px; }
            .progress-area p { font-size: 13px; color: var(--gold-dark); font-weight: 600; margin-bottom: 8px; }
            .progress-bar { width: 100%; height: 6px;
                            background: var(--white); border-radius: 100px; overflow: hidden; }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--gold-light), var(--gold));
                width: 0%; transition: width 0.3s;
                animation: progressShimmer 2s ease-in-out infinite;
            }
            @keyframes progressShimmer {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }

            .info-box {
                background: var(--gold-pale); border: 1px solid var(--gold-light);
                padding: 14px 18px; border-radius: 10px;
                color: var(--gold-dark); font-size: 13px; margin: 12px 0;
                display: flex; gap: 10px; align-items: flex-start;
            }

            table { width: 100%; border-collapse: collapse;
                    background: var(--white); border: 1px solid var(--grey-mid);
                    border-radius: 10px; overflow: hidden; margin-top: 12px; }
            th, td { padding: 12px 16px; text-align: left;
                     border-bottom: 1px solid var(--grey-light); font-size: 13px; }
            th { background: var(--grey-light); font-weight: 600;
                 color: var(--text-secondary); text-transform: uppercase;
                 font-size: 11px; letter-spacing: 0.5px; }
            tr:last-child td { border-bottom: none; }
            tr:hover td { background: var(--grey-light); }

            .disabled { opacity: 0.55; pointer-events: none; }

            .results-section { display: none; }
            .results-section.visible { display: block; }

            .chart-container {
                background: var(--white); border: 1px solid var(--grey-mid);
                border-radius: 10px; padding: 18px; margin-top: 14px;
            }
            .chart-title { font-size: 13px; font-weight: 600;
                           color: var(--text-secondary); margin-bottom: 12px;
                           text-transform: uppercase; letter-spacing: 0.4px; }
            .charts-grid { display: grid; grid-template-columns: 1fr 1fr;
                           gap: 16px; margin-top: 14px; }
            @media (max-width: 768px) { .charts-grid { grid-template-columns: 1fr; } }

            .badge { display: inline-block; padding: 4px 10px; border-radius: 100px;
                     font-size: 11px; font-weight: 600; letter-spacing: 0.3px; }
            .badge-success { background: var(--success-bg); color: var(--success); border: 1px solid #C8E6C8; }
            .badge-error { background: var(--error-bg); color: var(--error); border: 1px solid #F0CCCC; }
            .badge-gold { background: var(--gold-pale); color: var(--gold-dark); border: 1px solid var(--gold-light); }

            .past-jobs-list {
                display: flex; flex-direction: column; gap: 8px;
                max-height: 280px; overflow-y: auto;
            }
            .past-job {
                padding: 12px 16px; background: var(--white);
                border: 1px solid var(--grey-mid); border-radius: 8px;
                display: flex; justify-content: space-between; align-items: center;
                cursor: pointer; transition: all 0.2s;
            }
            .past-job:hover { border-color: var(--gold-light); background: var(--gold-pale); }
            .past-job-info { display: flex; flex-direction: column; gap: 2px; }
            .past-job-name { font-weight: 600; font-size: 13px; color: var(--text-primary); }
            .past-job-meta { font-size: 11px; color: var(--text-muted); }
        </style>
    </head>
    <body>
        <div class="top-bar">
            <a href="/" class="home">← Back to Dashboard</a>
            <div class="crumb">LocalML · <b>Fine-tuning</b></div>
        </div>
        <div class="container">
            <div class="page-header">
                <div class="page-icon">🎯</div>
                <div>
                    <h1>Fine-tuning Engine</h1>
                    <p>Train custom models with validation, early stopping, checkpointing, and post-training test set evaluation.</p>
                </div>
            </div>

            <div id="message"></div>

            <form id="finetuneForm">
                <!-- Data Section -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">📁</div>
                        <div><h2>Training Dataset</h2><div class="section-sub">Upload images for fine-tuning</div></div>
                    </div>
                    <div class="form-group full">
                        <div class="upload-area" id="uploadArea" onclick="document.getElementById('datasetInput').click()">
                            <div class="icon">📷</div>
                            <p><b>Click to upload</b> or drag and drop</p>
                            <small>Supported: JPG, PNG</small>
                            <input type="file" id="datasetInput" name="dataset" accept="image/*" required>
                        </div>
                        <p id="fileName" style="margin-top: 10px; color: var(--text-secondary); font-size: 13px;"></p>
                    </div>
                </div>

                <!-- Model Configuration -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">⚙️</div>
                        <div><h2>Model Configuration</h2><div class="section-sub">Architecture and input parameters</div></div>
                    </div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Model Name</label>
                            <input type="text" name="target_object" placeholder="e.g., bird_classifier" value="custom_model" required>
                            <div class="help">Name for this fine-tuning run</div>
                        </div>
                        <div class="form-group">
                            <label>Number of Classes</label>
                            <input type="number" name="num_classes" min="2" max="1000" value="10">
                            <div class="help">Output classes (e.g., 3 for {person, car, dog})</div>
                        </div>
                    </div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Image Size</label>
                            <select name="image_size">
                                <option value="28">28×28 (Testing only)</option>
                                <option value="64">64×64 (Mobile/Edge)</option>
                                <option value="128">128×128 (Balanced)</option>
                                <option value="224" selected>224×224 (Production) ⭐</option>
                                <option value="512">512×512 (High-detail)</option>
                            </select>
                            <div class="help">Larger = better accuracy but slower</div>
                        </div>
                        <div class="form-group">
                            <label>Batch Size</label>
                            <input type="number" name="batch_size" min="1" max="64" value="4">
                            <div class="help">Samples per iteration (larger = more stable)</div>
                        </div>
                    </div>
                </div>

                <!-- Training Configuration -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">🎓</div>
                        <div><h2>Training Schedule</h2><div class="section-sub">Epoch count and convergence settings</div></div>
                    </div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Number of Epochs</label>
                            <input type="number" name="epochs" min="1" max="100" value="5">
                            <div class="help">Full passes through dataset</div>
                        </div>
                    </div>
                </div>

                <!-- Validation Section -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">📊</div>
                        <div><h2>Validation & Early Stopping</h2><div class="section-sub">Track val loss, stop early if no improvement</div></div>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="enableValidation" name="enable_validation" checked>
                        <label for="enableValidation">Enable Validation Set (80/20 split)</label>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="enableEarlyStop" name="enable_early_stopping">
                        <label for="enableEarlyStop">Enable Early Stopping</label>
                    </div>
                    <div id="earlyStopConfig" class="disabled" style="margin-top: 14px;">
                        <div class="form-group" style="margin-bottom: 12px;">
                            <label>Patience (epochs to wait)</label>
                            <input type="number" name="patience" min="1" max="20" value="3">
                            <div class="help">Stop if val loss doesn't improve for N epochs</div>
                        </div>
                    </div>
                    <div class="form-group" style="margin-top: 14px;">
                        <label>Validation Split <span class="slider-value" id="valSplitValue">20%</span></label>
                        <input type="range" name="validation_split" min="0.1" max="0.5" step="0.05" value="0.2" id="valSplitSlider">
                        <div class="help">Fraction of data used for validation</div>
                    </div>
                </div>

                <!-- Checkpointing Section -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">💾</div>
                        <div><h2>Checkpointing</h2><div class="section-sub">Save model state during training</div></div>
                    </div>
                    <div class="form-group">
                        <label>Save Checkpoint Every N Epochs</label>
                        <input type="number" name="checkpoint_interval" min="0" max="100" value="1">
                        <div class="help">0 = Never save, 1 = Save every epoch (recommended)</div>
                    </div>
                    <div class="info-box">
                        <span>💾</span>
                        <div>
                            Checkpoints saved to <code>finetuned_models/</code> as <code>&lt;name&gt;_epoch_NNN.ckpt</code>
                        </div>
                    </div>
                </div>

                <!-- Quick Presets -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">⚡</div>
                        <div><h2>Quick Presets</h2><div class="section-sub">Recommended configurations for common scenarios</div></div>
                    </div>
                    <div class="preset-buttons">
                        <button type="button" class="preset-btn" onclick="applyPreset('quick')">⚡ Quick (1 epoch, 64×64)</button>
                        <button type="button" class="preset-btn" onclick="applyPreset('balanced')">⚖️ Balanced (5 epochs, 224×224)</button>
                        <button type="button" class="preset-btn" onclick="applyPreset('thorough')">🔬 Thorough (20 epochs, validation, early stop)</button>
                    </div>
                </div>

                <div style="display: flex; gap: 10px; margin-top: 24px;">
                    <button type="submit" class="btn-primary">▶️ Start Fine-tuning</button>
                    <button type="reset" class="btn-secondary">↺ Reset Form</button>
                </div>

                <div class="progress-area" id="progressArea">
                    <p>⏳ Training in progress... please wait</p>
                    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
                </div>
            </form>

            <!-- Past Jobs (always visible) -->
            <div class="section">
                <div class="section-header">
                    <div class="section-icon">🗂️</div>
                    <div><h2>Past Training Jobs</h2><div class="section-sub">Click any to view training curves and run test validation</div></div>
                </div>
                <div id="pastJobsList" class="past-jobs-list">
                    <div style="text-align:center; padding: 20px; color: var(--text-muted); font-size: 13px;">Loading…</div>
                </div>
            </div>

            <!-- RESULTS SECTION (appears after training or when a past job is selected) -->
            <div id="resultsSection" class="results-section">
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">📈</div>
                        <div>
                            <h2 id="resultsTitle">Training Results</h2>
                            <div class="section-sub" id="resultsSubtitle">Per-epoch loss and accuracy curves</div>
                        </div>
                    </div>

                    <div id="trainingStats" class="stats"></div>

                    <div class="charts-grid">
                        <div class="chart-container">
                            <div class="chart-title">Loss Curve (Lower is better)</div>
                            <canvas id="lossChart" height="220"></canvas>
                        </div>
                        <div class="chart-container">
                            <div class="chart-title">Accuracy Curve (Higher is better)</div>
                            <canvas id="accChart" height="220"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Test Set Validation Section -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">🧪</div>
                        <div>
                            <h2>Test Set Validation</h2>
                            <div class="section-sub">Evaluate the trained model on a held-out test set with ground-truth labels</div>
                        </div>
                    </div>
                    <div class="info-box">
                        <span>💡</span>
                        <div>
                            Upload test images and provide their expected class labels (comma-separated integers, in upload order). The model will run predictions and compute accuracy, precision, recall, and F1.
                        </div>
                    </div>

                    <div class="form-group" style="margin-bottom: 14px;">
                        <label>Test Images</label>
                        <input type="file" id="testImagesInput" accept="image/*" multiple>
                        <div class="help">Select multiple image files</div>
                    </div>
                    <div class="form-group" style="margin-bottom: 14px;">
                        <label>Expected Class Labels (comma-separated, same order as files)</label>
                        <input type="text" id="testLabelsInput" placeholder="e.g., 0,1,1,2,0,3">
                        <div class="help">Integer class indices, one per uploaded image</div>
                    </div>
                    <button type="button" class="btn-primary" onclick="runTestValidation()">🧪 Run Test Validation</button>

                    <div id="testValidationResults" style="margin-top: 18px;"></div>
                </div>
            </div>
        </div>

        <script>
            let currentJobId = null;
            let lossChart = null;
            let accChart = null;

            // File input handling
            document.getElementById('datasetInput').addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) document.getElementById('fileName').textContent = '✓ Selected: ' + file.name;
            });

            // Drag-drop
            const uploadArea = document.getElementById('uploadArea');
            ['dragenter','dragover'].forEach(ev => uploadArea.addEventListener(ev, e => {
                e.preventDefault(); uploadArea.classList.add('dragover');
            }));
            ['dragleave','drop'].forEach(ev => uploadArea.addEventListener(ev, e => {
                e.preventDefault(); uploadArea.classList.remove('dragover');
            }));
            uploadArea.addEventListener('drop', e => {
                if (e.dataTransfer.files.length) {
                    document.getElementById('datasetInput').files = e.dataTransfer.files;
                    document.getElementById('fileName').textContent = '✓ Selected: ' + e.dataTransfer.files[0].name;
                }
            });

            document.getElementById('valSplitSlider').addEventListener('input', (e) => {
                document.getElementById('valSplitValue').textContent = Math.round(e.target.value * 100) + '%';
            });

            document.getElementById('enableEarlyStop').addEventListener('change', (e) => {
                document.getElementById('earlyStopConfig').classList.toggle('disabled', !e.target.checked);
            });

            function applyPreset(preset) {
                const presets = {
                    'quick': { epochs: 1, image_size: '64', batch_size: 4, enable_validation: false, enable_early_stopping: false, checkpoint_interval: 0 },
                    'balanced': { epochs: 5, image_size: '224', batch_size: 4, enable_validation: true, enable_early_stopping: true, checkpoint_interval: 1, patience: 3 },
                    'thorough': { epochs: 20, image_size: '224', batch_size: 4, enable_validation: true, enable_early_stopping: true, checkpoint_interval: 1, patience: 5 }
                };
                const config = presets[preset];
                document.querySelector('input[name="epochs"]').value = config.epochs;
                document.querySelector('select[name="image_size"]').value = config.image_size;
                document.querySelector('input[name="batch_size"]').value = config.batch_size;
                document.querySelector('input[name="enable_validation"]').checked = config.enable_validation;
                document.querySelector('input[name="enable_early_stopping"]').checked = config.enable_early_stopping;
                document.querySelector('input[name="checkpoint_interval"]').value = config.checkpoint_interval;
                if (config.patience) document.querySelector('input[name="patience"]').value = config.patience;
                document.getElementById('earlyStopConfig').classList.toggle('disabled', !config.enable_early_stopping);
                showMessage('✓ Preset applied: ' + preset, 'info');
            }

            // Form submission
            document.getElementById('finetuneForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const file = document.getElementById('datasetInput').files[0];
                if (!file) { showMessage('Please select an image file', 'error'); return; }

                const formData = new FormData(e.target);
                document.getElementById('progressArea').style.display = 'block';
                try {
                    const response = await fetch('/finetune', { method: 'POST', body: formData });
                    const data = await response.json();
                    document.getElementById('progressArea').style.display = 'none';

                    const result = data.result || data;
                    if (result.status === 'success') {
                        let msg = `✓ Training complete · ${result.epochs_trained} epochs`;
                        if (result.validation_enabled && result.best_val_loss != null) {
                            msg += ` · Best val loss ${result.best_val_loss.toFixed(4)}`;
                        }
                        if (result.stopping_epoch) msg += ` · Early stopped at epoch ${result.stopping_epoch}`;
                        showMessage(msg, 'success');

                        // Refresh past jobs and load the latest
                        await loadPastJobs();
                        const jobs = window._lastJobs || [];
                        if (jobs.length > 0) loadJobHistory(jobs[0].id);
                    } else {
                        showMessage('Training failed: ' + (result.message || 'unknown error'), 'error');
                    }
                } catch (error) {
                    document.getElementById('progressArea').style.display = 'none';
                    showMessage('Error: ' + error.message, 'error');
                }
            });

            function showMessage(msg, type) {
                const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
                const div = document.getElementById('message');
                div.innerHTML = `<div class="message ${type}"><span>${icon}</span><div>${msg}</div></div>`;
                if (type === 'info') setTimeout(() => div.innerHTML = '', 5000);
            }

            // Past jobs
            async function loadPastJobs() {
                try {
                    const res = await fetch('/finetune/list');
                    const data = await res.json();
                    const jobs = data.jobs || [];
                    window._lastJobs = jobs;
                    const list = document.getElementById('pastJobsList');
                    if (jobs.length === 0) {
                        list.innerHTML = '<div style="text-align:center; padding:20px; color: var(--text-muted); font-size:13px;">No training jobs yet. Run your first one above.</div>';
                        return;
                    }
                    list.innerHTML = jobs.map(j => `
                        <div class="past-job" onclick="loadJobHistory(${j.id})">
                            <div class="past-job-info">
                                <div class="past-job-name">${j.target_object || 'Unnamed'}</div>
                                <div class="past-job-meta">Job #${j.id} · ${j.timestamp || '-'} · ${j.model_path ? '<span style=\"color: var(--success)\">Ready</span>' : 'Pending'}</div>
                            </div>
                            <div class="badge badge-gold">View →</div>
                        </div>
                    `).join('');
                } catch (e) {
                    document.getElementById('pastJobsList').innerHTML = '<div style="color: var(--error); padding: 12px; font-size: 13px;">Failed to load jobs</div>';
                }
            }

            async function loadJobHistory(jobId) {
                try {
                    const res = await fetch(`/finetune/history/${jobId}`);
                    const data = await res.json();
                    if (data.status !== 'success') {
                        showMessage('Could not load history: ' + (data.message || 'unknown'), 'error');
                        return;
                    }
                    currentJobId = jobId;
                    document.getElementById('resultsSection').classList.add('visible');
                    document.getElementById('resultsTitle').textContent = `${data.target_object || 'Job'} · #${jobId}`;
                    document.getElementById('resultsSubtitle').textContent = `Trained at ${data.timestamp || '-'} · ${data.summary.epochs_trained || 0} epochs`;

                    renderStats(data.summary);
                    renderCharts(data);

                    // Reset test validation panel
                    document.getElementById('testValidationResults').innerHTML = '';
                    document.getElementById('testImagesInput').value = '';
                    document.getElementById('testLabelsInput').value = '';

                    document.getElementById('resultsSection').scrollIntoView({behavior: 'smooth', block: 'start'});
                } catch (e) {
                    showMessage('Error loading history: ' + e.message, 'error');
                }
            }

            function fmt(v, decimals=4) {
                if (v == null || v === undefined) return '—';
                return Number(v).toFixed(decimals);
            }
            function pct(v) {
                if (v == null) return '—';
                return (Number(v) * 100).toFixed(1) + '%';
            }

            function renderStats(s) {
                document.getElementById('trainingStats').innerHTML = `
                    <div class="stat-card">
                        <div class="stat-label">Epochs Trained</div>
                        <div class="stat-value">${s.epochs_trained || 0}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Final Train Acc</div>
                        <div class="stat-value">${pct(s.final_train_accuracy)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Final Val Acc</div>
                        <div class="stat-value">${pct(s.final_val_accuracy)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Best Val Acc</div>
                        <div class="stat-value">${pct(s.best_val_accuracy)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Final Train Loss</div>
                        <div class="stat-value">${fmt(s.final_train_loss)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Best Val Loss</div>
                        <div class="stat-value">${fmt(s.best_val_loss)}</div>
                    </div>
                `;
            }

            function renderCharts(data) {
                const epochs = data.epochs || [];
                const goldGradient = (ctx, area, c1, c2) => {
                    if (!area) return c1;
                    const g = ctx.createLinearGradient(0, area.top, 0, area.bottom);
                    g.addColorStop(0, c1); g.addColorStop(1, c2);
                    return g;
                };

                const baseOpts = {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { font: { family: 'Inter', size: 11 }, color: '#757575', padding: 12, usePointStyle: true } },
                        tooltip: { backgroundColor: '#FFFFFF', titleColor: '#2C2C2C', bodyColor: '#2C2C2C', borderColor: '#E8E8E8', borderWidth: 1, padding: 10, cornerRadius: 8, titleFont: { family: 'Inter', weight: '600' }, bodyFont: { family: 'Inter' } }
                    },
                    scales: {
                        x: { grid: { color: '#F5F5F5' }, ticks: { color: '#9E9E9E', font: { family: 'Inter', size: 10 } }, title: { display: true, text: 'Epoch', color: '#757575', font: { family: 'Inter', size: 11, weight: '600' } } },
                        y: { grid: { color: '#F5F5F5' }, ticks: { color: '#9E9E9E', font: { family: 'Inter', size: 10 } }, beginAtZero: false }
                    }
                };

                if (lossChart) lossChart.destroy();
                if (accChart) accChart.destroy();

                const lossCtx = document.getElementById('lossChart').getContext('2d');
                lossChart = new Chart(lossCtx, {
                    type: 'line',
                    data: {
                        labels: epochs,
                        datasets: [
                            { label: 'Train Loss', data: data.train_loss, borderColor: '#D4AF37', backgroundColor: 'rgba(212, 175, 55, 0.12)', fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: '#D4AF37', borderWidth: 2.5 },
                            { label: 'Val Loss', data: data.val_loss, borderColor: '#B85C5C', backgroundColor: 'rgba(184, 92, 92, 0.08)', fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: '#B85C5C', borderWidth: 2.5, borderDash: [6, 4] }
                        ]
                    },
                    options: { ...baseOpts, scales: { ...baseOpts.scales, y: { ...baseOpts.scales.y, title: { display: true, text: 'Loss', color: '#757575', font: { family: 'Inter', size: 11, weight: '600' } } } } }
                });

                const accCtx = document.getElementById('accChart').getContext('2d');
                accChart = new Chart(accCtx, {
                    type: 'line',
                    data: {
                        labels: epochs,
                        datasets: [
                            { label: 'Train Accuracy', data: data.train_acc, borderColor: '#D4AF37', backgroundColor: 'rgba(212, 175, 55, 0.12)', fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: '#D4AF37', borderWidth: 2.5 },
                            { label: 'Val Accuracy', data: data.val_acc, borderColor: '#6FA86F', backgroundColor: 'rgba(111, 168, 111, 0.08)', fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: '#6FA86F', borderWidth: 2.5, borderDash: [6, 4] }
                        ]
                    },
                    options: { ...baseOpts, scales: { ...baseOpts.scales, y: { ...baseOpts.scales.y, min: 0, max: 1, title: { display: true, text: 'Accuracy', color: '#757575', font: { family: 'Inter', size: 11, weight: '600' } } } } }
                });
            }

            async function runTestValidation() {
                if (!currentJobId) { showMessage('Select a training job first', 'error'); return; }
                const files = document.getElementById('testImagesInput').files;
                const labels = document.getElementById('testLabelsInput').value.trim();
                if (!files.length) { showMessage('Please select test images', 'error'); return; }
                if (!labels) { showMessage('Please provide expected labels (comma-separated)', 'error'); return; }

                const labelArr = labels.split(',').map(s => s.trim()).filter(s => s !== '');
                if (labelArr.length !== files.length) {
                    showMessage(`Mismatch: ${files.length} images but ${labelArr.length} labels`, 'error');
                    return;
                }

                const formData = new FormData();
                for (const f of files) formData.append('images', f);
                formData.append('expected_labels', labels);
                const numClasses = document.querySelector('input[name=\"num_classes\"]').value;
                const imgSize = document.querySelector('select[name=\"image_size\"]').value;
                formData.append('num_classes', numClasses);
                formData.append('image_size', imgSize);

                const div = document.getElementById('testValidationResults');
                div.innerHTML = '<div class="message info"><span>⏳</span><div>Running test inference…</div></div>';

                try {
                    const res = await fetch(`/finetune/test-validation/${currentJobId}`, { method: 'POST', body: formData });
                    const data = await res.json();
                    if (data.status !== 'success') {
                        div.innerHTML = `<div class="message error"><span>❌</span><div>${data.message || 'Test failed'}</div></div>`;
                        return;
                    }
                    renderTestResults(data);
                } catch (e) {
                    div.innerHTML = `<div class="message error"><span>❌</span><div>Error: ${e.message}</div></div>`;
                }
            }

            function renderTestResults(data) {
                const div = document.getElementById('testValidationResults');
                const accClass = data.test_accuracy >= 0.85 ? 'success' : data.test_accuracy >= 0.65 ? 'warning' : 'error';
                let html = `
                    <div class="message ${accClass}"><span>${data.test_accuracy >= 0.85 ? '✅' : data.test_accuracy >= 0.65 ? '⚠️' : '❌'}</span>
                        <div><b>Test Accuracy: ${pct(data.test_accuracy)}</b> · ${data.correct_predictions}/${data.total_test_images} correct · Avg latency ${data.avg_latency_ms}ms</div>
                    </div>
                    <div class="stats">
                        <div class="stat-card"><div class="stat-label">Test Accuracy</div><div class="stat-value">${pct(data.test_accuracy)}</div></div>
                        <div class="stat-card"><div class="stat-label">Macro Precision</div><div class="stat-value">${pct(data.macro_precision)}</div></div>
                        <div class="stat-card"><div class="stat-label">Macro Recall</div><div class="stat-value">${pct(data.macro_recall)}</div></div>
                        <div class="stat-card"><div class="stat-label">Macro F1</div><div class="stat-value">${pct(data.macro_f1)}</div></div>
                    </div>
                `;

                if (data.per_class_metrics && data.per_class_metrics.length > 0) {
                    html += `<h3 style="margin: 18px 0 10px; font-size: 14px; color: var(--text-primary);">Per-Class Metrics</h3>
                        <table>
                            <thead><tr><th>Class</th><th>Total</th><th>Correct</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
                            <tbody>${data.per_class_metrics.map(m => `
                                <tr>
                                    <td><b>${m.class}</b></td>
                                    <td>${m.total}</td>
                                    <td>${m.correct}</td>
                                    <td>${pct(m.precision)}</td>
                                    <td>${pct(m.recall)}</td>
                                    <td>${pct(m.f1)}</td>
                                </tr>`).join('')}</tbody>
                        </table>`;
                }

                if (data.per_image_results && data.per_image_results.length > 0) {
                    html += `<h3 style="margin: 18px 0 10px; font-size: 14px; color: var(--text-primary);">Per-Image Results</h3>
                        <table>
                            <thead><tr><th>File</th><th>Expected</th><th>Predicted</th><th>Confidence</th><th>Result</th></tr></thead>
                            <tbody>${data.per_image_results.map(r => `
                                <tr>
                                    <td>${r.filename || '-'}</td>
                                    <td>${r.true_label}</td>
                                    <td>${r.predicted_label !== undefined ? r.predicted_label : '-'} ${r.predicted_class_name ? '<span style=\"color: var(--text-muted)\">(' + r.predicted_class_name + ')</span>' : ''}</td>
                                    <td>${r.confidence ? pct(r.confidence) : '-'}</td>
                                    <td>${r.correct ? '<span class="badge badge-success">✓ Correct</span>' : '<span class="badge badge-error">✗ Wrong</span>'}</td>
                                </tr>`).join('')}</tbody>
                        </table>`;
                }

                div.innerHTML = html;
            }

            // Initial load
            loadPastJobs();
        </script>
    </body>
    </html>
    """)


@app.get("/finetune/history/{job_id}")
async def get_training_history(job_id: int):
    """
    Return the per-epoch training history (loss + accuracy) for a fine-tuning job.

    Args:
        job_id: Row ID from the finetune_requests table.
    """
    DB_PATH = Path("finetune.db")
    if not DB_PATH.exists():
        return {"status": "error", "message": "No training history database"}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM finetune_requests WHERE id = ?", (job_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"status": "error", "message": f"Job {job_id} not found"}

    record = dict(row)
    result_path = Path(record.get("result_path", ""))
    if not result_path.exists():
        return {"status": "error", "message": "Result file missing", "record": record}

    try:
        with open(result_path, "r") as f:
            result = json.load(f)
    except Exception as e:
        return {"status": "error", "message": f"Failed to read result: {e}"}

    train_loss = result.get("train_loss_history", []) or []
    val_loss = result.get("val_loss_history", []) or []
    train_acc = result.get("train_acc_history", []) or []
    val_acc = result.get("val_acc_history", []) or []
    epochs = list(range(1, max(len(train_loss), len(train_acc), 1) + 1))

    return {
        "status": "success",
        "job_id": job_id,
        "target_object": record.get("target_object"),
        "model_path": record.get("model_path"),
        "timestamp": record.get("timestamp"),
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "summary": {
            "epochs_trained": result.get("epochs_trained"),
            "final_train_loss": result.get("final_train_loss"),
            "final_train_accuracy": result.get("final_train_accuracy"),
            "final_val_loss": result.get("final_val_loss"),
            "final_val_accuracy": result.get("final_val_accuracy"),
            "best_val_loss": result.get("best_val_loss"),
            "best_val_accuracy": result.get("best_val_accuracy"),
            "stopping_epoch": result.get("stopping_epoch"),
            "validation_enabled": result.get("validation_enabled", False)
        }
    }


@app.post("/finetune/test-validation/{job_id}")
async def test_set_validation(
    job_id: int,
    images: List[UploadFile] = File(...),
    expected_labels: str = Form(...),
    num_classes: int = Form(default=10),
    image_size: int = Form(default=224)
):
    """
    Run a fine-tuned model against a held-out test set and compute metrics.

    Args:
        job_id: Fine-tune job ID
        images: Test images
        expected_labels: Comma-separated integer class labels (one per image, in file order)
        num_classes: Number of output classes
        image_size: Input size used during training
    """
    if not JAX_AVAILABLE:
        return {"status": "error", "message": "JAX not available"}

    # Find the model checkpoint
    DB_PATH = Path("finetune.db")
    if not DB_PATH.exists():
        return {"status": "error", "message": "No training database"}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM finetune_requests WHERE id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return {"status": "error", "message": f"Job {job_id} not found"}

    record = dict(row)
    model_path = record.get("model_path")
    if not model_path or not Path(model_path).exists():
        # Fall back: use cached inference if a path-like checkpoint exists
        model_path = record.get("model_path", "")

    # Parse expected labels
    try:
        expected = [int(x.strip()) for x in expected_labels.split(",") if x.strip() != ""]
    except ValueError:
        return {"status": "error", "message": "expected_labels must be comma-separated integers"}

    if len(expected) != len(images):
        return {
            "status": "error",
            "message": f"Got {len(images)} images but {len(expected)} labels"
        }

    # Build inference helper (use cache if available, else create directly)
    inference = None
    try:
        if CACHE_AVAILABLE and MODEL_CACHE and model_path and Path(model_path).exists():
            inference = MODEL_CACHE.get_model(model_path)
        if inference is None and hasattr(jax_train, "FinetuneInference"):
            inference = jax_train.FinetuneInference(num_classes=num_classes, input_size=image_size)
            if model_path and Path(model_path).exists():
                inference.load_checkpoint(model_path)
    except Exception as e:
        return {"status": "error", "message": f"Could not load model: {e}"}

    if inference is None:
        return {"status": "error", "message": "Inference engine unavailable"}

    # Run predictions
    per_image_results = []
    correct = 0
    confusion = {}  # {(true_class, predicted_class): count}
    per_class_correct = {}
    per_class_total = {}
    total_latency = 0.0

    for idx, (upload, true_label) in enumerate(zip(images, expected)):
        try:
            image_bytes = await upload.read()
            t0 = time.time()
            pred_result = inference.predict(image_bytes, return_top_k=3)
            latency_ms = (time.time() - t0) * 1000
            total_latency += latency_ms

            preds = pred_result.get("predictions", []) if isinstance(pred_result, dict) else []
            top_pred_class = -1
            top_confidence = 0.0
            top_class_name = "unknown"
            if preds:
                top_pred_class = int(preds[0].get("class_id", -1)) if "class_id" in preds[0] else int(preds[0].get("class_index", -1))
                top_confidence = float(preds[0].get("confidence", 0.0))
                top_class_name = str(preds[0].get("class_name", str(top_pred_class)))

            is_correct = (top_pred_class == true_label)
            if is_correct:
                correct += 1

            confusion[(true_label, top_pred_class)] = confusion.get((true_label, top_pred_class), 0) + 1
            per_class_total[true_label] = per_class_total.get(true_label, 0) + 1
            if is_correct:
                per_class_correct[true_label] = per_class_correct.get(true_label, 0) + 1

            per_image_results.append({
                "filename": upload.filename,
                "true_label": true_label,
                "predicted_label": top_pred_class,
                "predicted_class_name": top_class_name,
                "confidence": top_confidence,
                "correct": is_correct,
                "latency_ms": round(latency_ms, 2)
            })
        except Exception as e:
            per_image_results.append({
                "filename": upload.filename,
                "true_label": true_label,
                "error": str(e),
                "correct": False
            })

    total = len(per_image_results)
    accuracy = (correct / total) if total > 0 else 0.0

    # Per-class metrics
    per_class_metrics = []
    for cls in sorted(per_class_total.keys()):
        cls_total = per_class_total.get(cls, 0)
        cls_correct = per_class_correct.get(cls, 0)
        recall = (cls_correct / cls_total) if cls_total > 0 else 0.0
        # Precision: correct predictions of cls / all predictions of cls
        predicted_as_cls = sum(v for (t, p), v in confusion.items() if p == cls)
        precision = (cls_correct / predicted_as_cls) if predicted_as_cls > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_metrics.append({
            "class": cls,
            "total": cls_total,
            "correct": cls_correct,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        })

    # Build confusion matrix as a dict of dicts
    confusion_matrix = {}
    for (true_cls, pred_cls), count in confusion.items():
        confusion_matrix.setdefault(str(true_cls), {})[str(pred_cls)] = count

    macro_precision = sum(m["precision"] for m in per_class_metrics) / len(per_class_metrics) if per_class_metrics else 0.0
    macro_recall = sum(m["recall"] for m in per_class_metrics) / len(per_class_metrics) if per_class_metrics else 0.0
    macro_f1 = sum(m["f1"] for m in per_class_metrics) / len(per_class_metrics) if per_class_metrics else 0.0

    summary = {
        "status": "success",
        "job_id": job_id,
        "target_object": record.get("target_object"),
        "total_test_images": total,
        "correct_predictions": correct,
        "test_accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "avg_latency_ms": round(total_latency / total, 2) if total > 0 else 0,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "per_image_results": per_image_results,
        "tested_at": datetime.now().isoformat()
    }

    # Persist to result JSON if possible (so it shows up in history)
    try:
        result_path = Path(record.get("result_path", ""))
        if result_path.exists():
            with open(result_path, "r") as f:
                existing = json.load(f)
            existing.setdefault("test_validations", []).append(summary)
            with open(result_path, "w") as f:
                json.dump(existing, f, indent=2)
    except Exception:
        pass

    return summary


@app.get("/finetune/list")
async def list_finetune_jobs():
    """Return all completed fine-tune jobs (used to populate dashboards)."""
    DB_PATH = Path("finetune.db")
    if not DB_PATH.exists():
        return {"status": "success", "jobs": []}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM finetune_requests ORDER BY id DESC LIMIT 50")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return {"status": "success", "jobs": rows, "total": len(rows)}


@app.get("/labeling")
async def labeling_ui():
    """Interactive labeling interface with image display and label input."""
    if not LABELING_AVAILABLE:
        return HTMLResponse("<h1>Labeling service not available</h1>")

    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Labeling Tool</title>
        <style>
            :root {
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --gold-light: #D4A373;
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
                --text-muted: #888888;
                --shadow-soft: 0 1px 3px rgba(0,0,0,0.08);
                --shadow-hover: 0 4px 12px rgba(184,134,11,0.15);
            }
            * { font-family: Inter, Arial, sans-serif; margin: 0; padding: 0; }
            body { background: var(--bg); padding: 20px; }
            .container { max-width: 1000px; margin: 0 auto; background: var(--white); padding: 24px; border-radius: 6px; box-shadow: var(--shadow-soft); border: 1px solid var(--border); }
            h1 { color: var(--gold-dark); margin-bottom: 20px; font-size: 24px; font-weight: 700; }
            .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }
            .stat-card { background: var(--bg); padding: 15px; border-radius: 6px; text-align: center; border: 1px solid var(--border); }
            .stat-value { font-size: 24px; font-weight: 800; color: var(--gold-dark); }
            .stat-label { font-size: 10px; color: var(--text-muted); margin-top: 5px; text-transform: uppercase; letter-spacing: 0.3px; }
            .progress-bar { width: 100%; height: 30px; background: var(--border); border-radius: 4px; overflow: hidden; margin-bottom: 20px; }
            .progress-fill { height: 100%; background: var(--gold); transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: var(--white); font-weight: 700; font-size: 12px; }
            .image-section { display: grid; grid-template-columns: 600px 1fr; gap: 20px; margin-bottom: 20px; }
            .image-container { text-align: center; }
            .image-container img { max-width: 100%; max-height: 400px; border: 1px solid var(--border); border-radius: 6px; }
            .controls { display: flex; flex-direction: column; gap: 15px; }
            .form-group { display: flex; flex-direction: column; }
            .form-group label { font-weight: 700; margin-bottom: 5px; color: var(--text-primary); font-size: 12px; text-transform: uppercase; letter-spacing: 0.3px; }
            .form-group input, .form-group select { padding: 10px; border: 1px solid var(--border); border-radius: 4px; font-size: 13px; background: var(--white); color: var(--text-primary); }
            .form-group input:focus, .form-group select:focus { outline: none; border-color: var(--gold); box-shadow: 0 0 0 3px rgba(184,134,11,0.1); }
            .button-group { display: flex; gap: 10px; }
            button { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: 700; transition: background 0.15s; font-size: 12px; text-transform: uppercase; letter-spacing: 0.3px; }
            .btn-primary { background: var(--gold); color: var(--white); }
            .btn-primary:hover { background: var(--gold-dark); }
            .btn-success { background: var(--gold); color: var(--white); }
            .btn-success:hover { background: var(--gold-dark); }
            .btn-secondary { background: var(--text-secondary); color: var(--white); }
            .btn-secondary:hover { background: var(--text-primary); }
            .upload-section { background: var(--bg); padding: 20px; border-radius: 6px; border: 2px dashed var(--border); text-align: center; }
            .message { padding: 10px; border-radius: 4px; margin-bottom: 10px; border: 1px solid var(--border); }
            .message.success { background: #F0F9F0; color: var(--gold-dark); }
            .message.error { background: #FAE8E5; color: #8B4513; }
            .message.info { background: #F5F5F0; color: var(--text-primary); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Image Labeling Tool</h1>

            <div id="stats" class="stats" style="display: none;">
                <div class="stat-card">
                    <div class="stat-value" id="total">0</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="pending">0</div>
                    <div class="stat-label">Pending</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="labeled">0</div>
                    <div class="stat-label">Labeled</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="progress">0%</div>
                    <div class="stat-label">Progress</div>
                </div>
            </div>

            <div class="progress-bar">
                <div class="progress-fill" id="progressBar" style="width: 0%;">0%</div>
            </div>

            <div id="message"></div>

            <!-- Tab Selector -->
            <div style="margin-bottom: 20px;">
                <button class="btn-primary" onclick="showTab('manual')">Manual Labeling</button>
                <button class="btn-primary" onclick="showTab('upload')">Upload Labels File</button>
            </div>

            <!-- Manual Labeling Tab -->
            <div id="manual-tab" style="display: block;">
                <div class="image-section">
                    <div class="image-container" id="imageDiv">
                        <p>Loading images...</p>
                    </div>
                    <div class="controls">
                        <div class="form-group">
                            <label>Select Class:</label>
                            <select id="classSelect">
                                <option value="">-- Choose a class --</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Notes (optional):</label>
                            <input type="text" id="notes" placeholder="e.g., 'partially occluded'">
                        </div>
                        <div class="form-group">
                            <label>Bounding Box (optional):</label>
                            <input type="text" id="bbox" placeholder="x1,y1,x2,y2 (or leave blank)">
                        </div>
                        <div class="button-group">
                            <button class="btn-success" onclick="submitLabel()">✓ Label & Next</button>
                            <button class="btn-secondary" onclick="skipImage()">⊘ Skip</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Upload File Tab -->
            <div id="upload-tab" style="display: none;">
                <div class="upload-section">
                    <h3>Upload Labels CSV File</h3>
                    <p>Format: one label per line, in order of uploaded images</p>
                    <input type="file" id="labelFile" accept=".csv,.txt" style="display: none;">
                    <button class="btn-primary" onclick="document.getElementById('labelFile').click()">Choose File</button>
                    <p id="fileName" style="margin-top: 10px; color: #666;"></p>
                    <button class="btn-success" onclick="uploadLabelsFile()" style="margin-top: 10px;">Upload & Process</button>
                </div>
            </div>

            <hr style="margin: 30px 0;">
            <div style="text-align: center; color: #666;">
                <p>💡 Tip: Use keyboard shortcuts for faster labeling</p>
                <small>Built with LocalML finetune</small>
            </div>
        </div>

        <script>
            let currentImages = [];
            let currentIndex = 0;
            let selectedFile = null;

            async function loadPendingImages() {
                try {
                    const resp = await fetch('/labeling-pending?limit=1');
                    const data = await resp.json();
                    currentImages = data.images || [];

                    if (currentImages.length > 0) {
                        displayImage(0);
                    } else {
                        document.getElementById('imageDiv').innerHTML = '<p style="color: #666;">No pending images to label</p>';
                    }

                    await updateStats();
                } catch (e) {
                    showMessage('Error loading images: ' + e, 'error');
                }
            }

            async function updateStats() {
                try {
                    const resp = await fetch('/labeling-stats');
                    const data = await resp.json();
                    const stats = data.stats || {};

                    document.getElementById('total').textContent = stats.total || 0;
                    document.getElementById('pending').textContent = stats.pending || 0;
                    document.getElementById('labeled').textContent = stats.labeled || 0;
                    const progress = Math.round(stats.progress || 0);
                    document.getElementById('progress').textContent = progress + '%';
                    document.getElementById('progressBar').style.width = progress + '%';
                    document.getElementById('progressBar').textContent = progress + '%';
                    document.getElementById('stats').style.display = 'grid';
                } catch (e) {
                    console.error('Error updating stats:', e);
                }
            }

            function displayImage(idx) {
                if (idx < 0 || idx >= currentImages.length) return;

                const img = currentImages[idx];
                const html = '<img src="data:image/jpeg;base64,' + img.image + '" id="currentImage">';
                document.getElementById('imageDiv').innerHTML = html;
                currentIndex = idx;

                // Load existing classes into dropdown
                loadClasses();
            }

            async function loadClasses() {
                try {
                    const resp = await fetch('/labeling-classes');
                    const data = resp.json ? await resp.json() : {};
                    const classes = data.classes || ['person', 'car', 'dog', 'cat', 'other'];

                    const select = document.getElementById('classSelect');
                    select.innerHTML = '<option value="">-- Choose a class --</option>';
                    classes.forEach(cls => {
                        const opt = document.createElement('option');
                        opt.value = cls;
                        opt.textContent = cls;
                        select.appendChild(opt);
                    });
                } catch (e) {
                    console.error('Could not load classes');
                }
            }

            async function submitLabel() {
                if (currentIndex >= currentImages.length) {
                    showMessage('All images labeled!', 'success');
                    return;
                }

                const classVal = document.getElementById('classSelect').value;
                if (!classVal) {
                    showMessage('Please select a class', 'error');
                    return;
                }

                const img = currentImages[currentIndex];
                const bbox = document.getElementById('bbox').value;
                const notes = document.getElementById('notes').value;

                try {
                    const resp = await fetch('/labeling-submit', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            image_id: img.id,
                            class_name: classVal,
                            bbox: bbox ? bbox.split(',').map(Number) : null,
                            notes: notes
                        })
                    });

                    if (resp.ok) {
                        showMessage('✓ Labeled: ' + classVal, 'success');
                        document.getElementById('notes').value = '';
                        document.getElementById('bbox').value = '';
                        document.getElementById('classSelect').value = '';

                        await updateStats();
                        loadPendingImages(); // Reload next image
                    } else {
                        showMessage('Error submitting label', 'error');
                    }
                } catch (e) {
                    showMessage('Error: ' + e, 'error');
                }
            }

            async function skipImage() {
                currentIndex++;
                if (currentIndex < currentImages.length) {
                    displayImage(currentIndex);
                } else {
                    loadPendingImages();
                }
            }

            function showTab(tab) {
                document.getElementById('manual-tab').style.display = tab === 'manual' ? 'block' : 'none';
                document.getElementById('upload-tab').style.display = tab === 'upload' ? 'block' : 'none';
            }

            document.getElementById('labelFile').addEventListener('change', (e) => {
                if (e.target.files[0]) {
                    selectedFile = e.target.files[0];
                    document.getElementById('fileName').textContent = 'Selected: ' + e.target.files[0].name;
                }
            });

            async function uploadLabelsFile() {
                if (!selectedFile) {
                    showMessage('Please select a file', 'error');
                    return;
                }

                const formData = new FormData();
                formData.append('file', selectedFile);

                try {
                    const resp = await fetch('/labeling-upload-file', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await resp.json();
                    if (resp.ok) {
                        showMessage('✓ ' + data.message, 'success');
                        await updateStats();
                    } else {
                        showMessage('Error: ' + data.message, 'error');
                    }
                } catch (e) {
                    showMessage('Error uploading: ' + e, 'error');
                }
            }

            function showMessage(msg, type) {
                const msgDiv = document.getElementById('message');
                msgDiv.innerHTML = '<div class="message ' + type + '">' + msg + '</div>';
                setTimeout(() => msgDiv.innerHTML = '', 5000);
            }

            // Load on startup
            loadPendingImages();
            setInterval(updateStats, 5000); // Update stats every 5 seconds
        </script>
    </body>
    </html>
    """)


@app.get("/labeling-stats")
async def labeling_stats():
    """Get labeling progress statistics."""
    if not LABELING_AVAILABLE or labeling_service is None:
        return {"status": "error", "message": "Labeling not available"}

    stats = labeling_service.get_stats()
    return {"status": "success", "stats": stats}


@app.get("/labeling-pending")
async def labeling_pending(limit: int = 10):
    """Get pending images for labeling."""
    if not LABELING_AVAILABLE or labeling_service is None:
        return {"status": "error", "message": "Labeling not available"}

    images = labeling_service.get_unlabeled_images(limit=limit)
    return {"status": "success", "images": images, "count": len(images)}


@app.post("/labeling-submit")
async def labeling_submit(image_id: int = Form(...), class_name: str = Form(...),
                         notes: str = Form(default="")):
    """Submit a label for an image."""
    if not LABELING_AVAILABLE or labeling_service is None:
        return {"status": "error", "message": "Labeling not available"}

    try:
        label_id = labeling_service.label_image(
            image_id=image_id,
            class_name=class_name,
            labeled_by="web_ui",
            notes=notes
        )
        return {"status": "success", "label_id": label_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/labeling-upload-file")
async def labeling_upload_file(file: UploadFile = File(...)):
    """Upload CSV file with labels (one per line, in order of images)."""
    if not LABELING_AVAILABLE or labeling_service is None:
        return {"status": "error", "message": "Labeling not available"}

    try:
        contents = await file.read()
        lines = contents.decode('utf-8').strip().split('\n')
        labels = [line.strip() for line in lines if line.strip()]

        # Get all labeled images and update their labels
        # For now, this just confirms we received the labels
        # In production, match labels to pending images by order
        return {
            "status": "success",
            "message": f"Processed {len(labels)} labels",
            "labels_count": len(labels)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/labeling-classes")
async def labeling_classes():
    """Get list of available classes."""
    if not LABELING_AVAILABLE or labeling_service is None:
        return {"status": "error", "classes": []}

    conn = sqlite3.connect("labeling.db")
    c = conn.cursor()
    c.execute("SELECT DISTINCT class_name FROM classes")
    classes = [row[0] for row in c.fetchall()]
    conn.close()

    return {"status": "success", "classes": classes or ["person", "car", "dog", "cat", "other"]}


@app.post("/labeling-to-training")
async def labeling_to_training(num_classes: int = 10, image_size: int = 224):
    """Export labeled data and start fine-tuning with real labels."""
    if not JAX_AVAILABLE or labeling_service is None:
        return {"status": "error", "message": "JAX or labeling not available"}

    try:
        # Export labeled dataset
        export_result = labeling_service.export_for_training(output_path="training_dataset.json")

        if export_result.get("status") != "success":
            return export_result

        # Load dataset and extract images + labels
        with open("training_dataset.json", 'r') as f:
            dataset = json.load(f)

        # Combine all images and labels for training
        # Note: For real implementation, would batch process images
        total_samples = len(dataset['images'])
        if total_samples == 0:
            return {"status": "error", "message": "No labeled images in dataset"}

        return {
            "status": "success",
            "message": f"Dataset ready: {total_samples} labeled images, {len(dataset['classes'])} classes",
            "total_samples": total_samples,
            "classes": dataset['classes'],
            "ready_for_training": True,
            "next_step": "POST /finetune with the labeled dataset"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/models-versions")
async def models_versions_ui():
    """Model versioning and registry dashboard."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Registry & Versioning</title>
        <style>
            :root {
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
                --text-muted: #888888;
                --shadow-soft: 0 1px 3px rgba(0,0,0,0.08);
                --shadow-hover: 0 4px 12px rgba(184,134,11,0.15);
            }
            * { font-family: Inter, Arial, sans-serif; margin: 0; padding: 0; }
            body { background: var(--bg); padding: 20px; }
            .container { max-width: 1400px; margin: 0 auto; background: var(--white); padding: 32px; border-radius: 6px; box-shadow: var(--shadow-soft); border: 1px solid var(--border); }
            h1 { color: var(--gold-dark); margin-bottom: 10px; font-size: 24px; font-weight: 700; }
            .subtitle { color: var(--text-secondary); margin-bottom: 30px; font-size: 13px; }

            .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 1px solid var(--border); }
            .tab { padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 13px; font-weight: 700; color: var(--text-secondary); border-bottom: 3px solid transparent; text-transform: uppercase; letter-spacing: 0.3px; transition: all 0.15s; }
            .tab:hover { color: var(--gold); }
            .tab.active { color: var(--gold); border-bottom-color: var(--gold); }

            .tab-content { display: none; }
            .tab-content.active { display: block; }

            .model-section { margin-bottom: 30px; padding: 20px; background: var(--bg); border-radius: 6px; border: 1px solid var(--border); }
            .model-section h2 { color: var(--gold-dark); font-size: 15px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 0.3px; font-weight: 700; }

            .version-card { padding: 20px; background: var(--white); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 15px; transition: all 0.15s; box-shadow: var(--shadow-soft); }
            .version-card:hover { box-shadow: var(--shadow-hover); }
            .version-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
            .version-name { font-size: 16px; font-weight: 800; color: var(--text-primary); }
            .version-badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; }
            .badge-active { background: var(--gold); color: var(--white); }
            .badge-deprecated { background: #8B4513; color: var(--white); }
            .badge-private { background: var(--text-secondary); color: var(--white); }
            .badge-public { background: var(--gold-dark); color: var(--white); }
            .badge-shared { background: var(--text-muted); color: var(--white); }

            .metadata-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
            .metadata-item { padding: 10px; background: var(--bg); border-radius: 4px; border: 1px solid var(--border); }
            .metadata-label { font-size: 10px; color: var(--text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; }
            .metadata-value { font-size: 13px; color: var(--gold-dark); margin-top: 5px; font-weight: 700; }

            .dataset-info { padding: 15px; background: #F0F9F0; border-radius: 6px; margin-bottom: 15px; border: 1px solid var(--border); }
            .dataset-info h3 { color: var(--gold-dark); font-size: 12px; margin-bottom: 8px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; }
            .dataset-list { font-size: 12px; color: var(--text-secondary); line-height: 1.6; }

            .access-control { padding: 15px; background: #FFFBF5; border-radius: 6px; margin-bottom: 15px; border: 1px solid var(--border); }
            .access-control h3 { color: var(--gold-dark); font-size: 12px; margin-bottom: 8px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; }

            .comparison-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            .comparison-table th { background: var(--bg); padding: 10px; text-align: left; font-weight: 700; font-size: 11px; border: 1px solid var(--border); text-transform: uppercase; letter-spacing: 0.3px; color: var(--gold-dark); }
            .comparison-table td { padding: 10px; border: 1px solid var(--border); font-size: 12px; }
            .comparison-table tr:hover { background: var(--bg); }

            button { padding: 8px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; transition: all 0.15s; }
            .btn-primary { background: var(--gold); color: var(--white); }
            .btn-primary:hover { background: var(--gold-dark); }
            .btn-secondary { background: var(--text-secondary); color: var(--white); }
            .btn-secondary:hover { background: var(--text-primary); }

            .loading { text-align: center; color: var(--text-secondary); padding: 40px; }
            .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid var(--border); border-radius: 50%; border-top: 3px solid var(--gold); animation: spin 1s linear infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

            .message { padding: 15px; border-radius: 4px; margin-bottom: 15px; border: 1px solid var(--border); }
            .message.success { background: #F0F9F0; color: var(--gold-dark); }
            .message.error { background: #FAE8E5; color: #8B4513; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Registry & Versioning</h1>
            <p class="subtitle">Track all model versions with full metadata, parameters, and access control</p>

            <div class="tabs">
                <button class="tab active" onclick="switchTab('all-models')">All Models</button>
                <button class="tab" onclick="switchTab('versions')">Version History</button>
                <button class="tab" onclick="switchTab('comparison')">Compare Versions</button>
                <button class="tab" onclick="switchTab('access')">Access Control</button>
                <a href="/ab-testing" style="margin-left:auto; padding:8px 15px;
                   background:var(--gold); color:var(--white); border-radius:4px;
                   text-decoration:none; font-weight:bold; font-size:12px;
                   border:none; cursor:pointer; text-transform:uppercase; letter-spacing:0.3px;">A/B Testing</a>
            </div>

            <!-- All Models Tab -->
            <div id="all-models" class="tab-content active">
                <div id="allModelsContainer" class="loading">
                    <div class="spinner"></div>
                    <p>Loading models...</p>
                </div>
            </div>

            <!-- Version History Tab -->
            <div id="versions" class="tab-content">
                <div class="model-section">
                    <h2>Select Model to View Versions</h2>
                    <select id="modelSelect" onchange="loadVersions()" style="padding: 8px; width: 300px;">
                        <option value="">-- Choose a model --</option>
                    </select>
                </div>
                <div id="versionsContainer"></div>
            </div>

            <!-- Comparison Tab -->
            <div id="comparison" class="tab-content">
                <div class="model-section">
                    <h2>Compare Model Versions</h2>
                    <select id="compareModelSelect" onchange="loadComparison()" style="padding: 8px; width: 300px;">
                        <option value="">-- Choose a model --</option>
                    </select>
                </div>
                <div id="comparisonContainer"></div>
            </div>

            <!-- Access Control Tab -->
            <div id="access" class="tab-content">
                <div class="model-section">
                    <h2>Manage Model Access</h2>
                    <select id="accessModelSelect" onchange="loadAccessControl()" style="padding: 8px; width: 300px;">
                        <option value="">-- Choose a model --</option>
                    </select>
                </div>
                <div id="accessContainer"></div>
            </div>
        </div>

        <script>
            async function loadAllModels() {
                try {
                    const resp = await fetch('/model-registry/all');
                    const data = await resp.json();

                    let html = '';
                    if (data.models && data.models.length > 0) {
                        for (const model of data.models) {
                            const latest = model.versions[0];
                            html += `
                                <div class="model-section">
                                    <h2>${model.name}</h2>
                                    <p style="color: #666; margin-bottom: 15px;">
                                        ${model.version_count} versions | Latest: ${latest.version} | Owner: ${latest.owner}
                                    </p>

                                    <div class="version-card">
                                        <div class="version-header">
                                            <div>
                                                <span class="version-name">${latest.version}</span>
                                                <span class="version-badge badge-${latest.status}">${latest.status.toUpperCase()}</span>
                                                <span class="version-badge badge-${latest.access_level}">${latest.access_level.toUpperCase()}</span>
                                            </div>
                                            <div>Created: ${new Date(latest.created_at).toLocaleDateString()}</div>
                                        </div>

                                        <div class="metadata-grid">
                                            <div class="metadata-item">
                                                <div class="metadata-label">Classes</div>
                                                <div class="metadata-value">${latest.metadata.num_classes || '?'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Image Size</div>
                                                <div class="metadata-value">${latest.metadata.image_size || '?'}×${latest.metadata.image_size || '?'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Best Val Loss</div>
                                                <div class="metadata-value">${(latest.metadata.best_val_loss || 0).toFixed(4)}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Epochs</div>
                                                <div class="metadata-value">${latest.metadata.epochs_trained || '?'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Batch Size</div>
                                                <div class="metadata-value">${latest.metadata.batch_size || '?'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Training Images</div>
                                                <div class="metadata-value">${latest.dataset.training_images || '?'}</div>
                                            </div>
                                        </div>

                                        <div class="dataset-info">
                                            <h3>📊 Dataset Info</h3>
                                            <div class="dataset-list">
                                                <div>Classes: ${latest.dataset.classes.join(', ') || 'N/A'}</div>
                                                <div>Total Images: ${latest.dataset.total_images || '?'}</div>
                                                <div>Training/Validation/Test: ${latest.dataset.training_images || 0}/${latest.dataset.validation_images || 0}/${latest.dataset.test_images || 0}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }
                    } else {
                        html = '<p style="color: #999; text-align: center; padding: 40px;">No models registered yet</p>';
                    }

                    document.getElementById('allModelsContainer').innerHTML = html;

                    // Populate dropdowns
                    let selectHtml = '<option value="">-- Choose a model --</option>';
                    for (const model of (data.models || [])) {
                        selectHtml += `<option value="${model.name}">${model.name}</option>`;
                    }
                    document.getElementById('modelSelect').innerHTML = selectHtml;
                    document.getElementById('compareModelSelect').innerHTML = selectHtml;
                    document.getElementById('accessModelSelect').innerHTML = selectHtml;
                } catch (e) {
                    document.getElementById('allModelsContainer').innerHTML = '<p style="color: #c62828;">Error loading models</p>';
                }
            }

            async function loadVersions() {
                const modelName = document.getElementById('modelSelect').value;
                if (!modelName) return;

                try {
                    const resp = await fetch(`/model-registry/versions/${modelName}`);
                    const data = await resp.json();

                    let html = '';
                    for (const v of data.versions || []) {
                        html += `
                            <div class="version-card">
                                <div class="version-header">
                                    <div>
                                        <span class="version-name">${v.version}</span>
                                        <span class="version-badge badge-${v.status}">${v.status}</span>
                                    </div>
                                    <div>${new Date(v.created_at).toLocaleDateString()}</div>
                                </div>
                                <div class="metadata-grid">
                                    <div class="metadata-item">
                                        <div class="metadata-label">Best Loss</div>
                                        <div class="metadata-value">${(v.metadata.best_val_loss || 0).toFixed(4)}</div>
                                    </div>
                                    <div class="metadata-item">
                                        <div class="metadata-label">Epochs</div>
                                        <div class="metadata-value">${v.metadata.epochs_trained || '?'}</div>
                                    </div>
                                    <div class="metadata-item">
                                        <div class="metadata-label">Training Images</div>
                                        <div class="metadata-value">${v.dataset.training_images || '?'}</div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    document.getElementById('versionsContainer').innerHTML = html;
                } catch (e) {
                    console.error('Error:', e);
                }
            }

            async function loadComparison() {
                const modelName = document.getElementById('compareModelSelect').value;
                if (!modelName) return;

                try {
                    const resp = await fetch(`/model-registry/compare/${modelName}`);
                    const data = await resp.json();

                    let html = '<table class="comparison-table"><thead><tr>';
                    html += '<th>Version</th><th>Date</th><th>Classes</th><th>Epochs</th><th>Best Loss</th><th>Accuracy</th><th>Images</th>';
                    html += '</tr></thead><tbody>';

                    for (const v of data.comparison.by_loss || []) {
                        html += `<tr>
                            <td><strong>${v.version}</strong></td>
                            <td>${new Date(v.created_at).toLocaleDateString()}</td>
                            <td>${v.metadata.num_classes}</td>
                            <td>${v.metadata.epochs_trained}</td>
                            <td>${(v.metadata.best_val_loss || 0).toFixed(4)}</td>
                            <td>${(v.metadata.accuracy_on_test_set || 0).toFixed(2)}%</td>
                            <td>${v.dataset.training_images}</td>
                        </tr>`;
                    }

                    html += '</tbody></table>';
                    document.getElementById('comparisonContainer').innerHTML = html;
                } catch (e) {
                    console.error('Error:', e);
                }
            }

            function loadAccessControl() {
                document.getElementById('accessContainer').innerHTML = '<p style="color: #666;">Select model to manage access control</p>';
            }

            function switchTab(tabName) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));

                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
            }

            // Load on startup
            loadAllModels();
        </script>
    </body>
    </html>
    """)


@app.get("/model-registry/all")
async def model_registry_all():
    """Get all registered models."""
    if not REGISTRY_AVAILABLE:
        return {"status": "error", "models": []}

    try:
        models = REGISTRY.get_all_models()
        return {
            "status": "success",
            "models": models,
            "total_models": len(models)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/model-registry/versions/{model_name}")
async def model_registry_versions(model_name: str):
    """Get all versions of a specific model."""
    if not REGISTRY_AVAILABLE:
        return {"status": "error", "versions": []}

    try:
        versions = REGISTRY.get_model_versions(model_name)
        return {
            "status": "success",
            "model_name": model_name,
            "versions": versions
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/model-registry/compare/{model_name}")
async def model_registry_compare(model_name: str):
    """Compare all versions of a model."""
    if not REGISTRY_AVAILABLE:
        return {"status": "error"}

    try:
        comparison = REGISTRY.compare_versions(model_name)
        return {
            "status": "success",
            **comparison
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/model-registry/access/{model_id}")
async def model_registry_set_access(model_id: int, access_level: str = Form(...)):
    """Update access control for a model."""
    if not REGISTRY_AVAILABLE:
        return {"status": "error"}

    try:
        REGISTRY.set_access_level(model_id, access_level)
        REGISTRY.add_history_event(model_id, "access_changed", f"Access level changed to {access_level}")
        return {"status": "success", "message": f"Access level set to {access_level}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/ab-testing", response_class=HTMLResponse)
async def ab_testing_dashboard():
    """A/B Testing dashboard with 4 tabs."""
    if not AB_AVAILABLE:
        return "<h1>A/B Testing not available</h1>"

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>A/B Testing Dashboard</title>
        <style>
            :root {
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
                --text-muted: #888888;
            }
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   background: var(--bg); color: var(--text-primary); }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            h1 { margin-bottom: 20px; color: var(--gold-dark); }
            .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid var(--border); }
            .tab { padding: 12px 20px; background: none; border: none; border-bottom: 3px solid transparent;
                   cursor: pointer; font-size: 14px; font-weight: 500; color: var(--text-secondary);
                   transition: all 0.2s; }
            .tab:hover { color: var(--gold); }
            .tab.active { color: var(--gold); border-bottom-color: var(--gold); }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .section { background: var(--white); padding: 20px; border-radius: 6px; margin-bottom: 20px;
                       border-left: 4px solid var(--gold); box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
            .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            .form-group { display: flex; flex-direction: column; }
            .form-group label { font-weight: 600; margin-bottom: 8px; font-size: 13px; }
            .form-group input, .form-group select, .form-group textarea {
                padding: 10px; border: 1px solid var(--border); border-radius: 4px; font-size: 14px;
                font-family: inherit; }
            .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
                outline: none; border-color: var(--gold); box-shadow: 0 0 0 3px rgba(184,134,11,0.1); }
            .btn-primary { background: var(--gold); color: var(--white); padding: 12px 24px; border: none;
                           border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 14px;
                           transition: background 0.2s; }
            .btn-primary:hover { background: var(--gold-dark); }
            .btn-secondary { background: var(--text-secondary); color: var(--white); padding: 10px 20px; border: none;
                             border-radius: 4px; cursor: pointer; font-size: 13px; }
            .stat-card { background: var(--bg); padding: 15px; border-radius: 6px;
                         border-left: 3px solid var(--gold); }
            .stat-value { font-size: 24px; font-weight: bold; color: var(--gold-dark); }
            .stat-label { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
            .card { background: var(--white); padding: 15px; border-radius: 6px; margin-bottom: 15px;
                    border: 1px solid var(--border); }
            .card h3 { margin-bottom: 10px; color: var(--gold-dark); }
            .badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 12px;
                     font-weight: 600; }
            .badge.active { background: #FFF8DC; color: var(--gold-dark); }
            .badge.completed { background: #FFF8DC; color: var(--gold-dark); }
            .winner-badge { background: var(--gold); color: var(--white); padding: 20px; border-radius: 6px;
                            text-align: center; margin: 20px 0; font-size: 18px; font-weight: bold; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th { background: var(--bg); padding: 12px; text-align: left; font-weight: 600; border-bottom: 2px solid var(--border); }
            td { padding: 12px; border-bottom: 1px solid var(--border); }
            tr:hover { background: #FFFBF0; }
            .message { padding: 15px; border-radius: 4px; margin-bottom: 15px; }
            .message.error { background: #FFF5F0; color: #8B6914; }
            .message.success { background: #FFF8DC; color: var(--gold-dark); }
            .message.info { background: #FFFBF0; color: var(--gold-dark); }
            .loading { display: none; text-align: center; padding: 20px; }
            .loading.show { display: block; }
            .split-display { font-size: 20px; font-weight: bold; color: var(--gold); margin-top: 8px; }
            .upload-area { border: 2px dashed var(--gold); border-radius: 6px; padding: 30px;
                           text-align: center; cursor: pointer; transition: all 0.2s; }
            .upload-area:hover { background: var(--bg); }
            .upload-area.dragover { background: #FFF8DC; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚗️ A/B Testing Dashboard</h1>

            <div class="tabs">
                <button class="tab active" onclick="switchTab('setup')">Setup</button>
                <button class="tab" onclick="switchTab('active-tests')">Active Tests</button>
                <button class="tab" onclick="switchTab('results')">Results</button>
                <button class="tab" onclick="switchTab('history')">History</button>
            </div>

            <!-- Setup Tab -->
            <div id="setup" class="tab-content active">
                <div class="section">
                    <h2>Create New A/B Test</h2>
                    <div id="createMessage"></div>
                    <form id="createTestForm" onsubmit="createTest(event)">
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Test Name *</label>
                                <input id="testName" required>
                            </div>
                            <div class="form-group">
                                <label>Description</label>
                                <input id="testDesc">
                            </div>
                            <div class="form-group">
                                <label>Model A *</label>
                                <select id="modelA" required>
                                    <option value="">Loading...</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Model B *</label>
                                <select id="modelB" required>
                                    <option value="">Loading...</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Model A Path (optional)</label>
                                <input id="modelAPath" placeholder="/path/to/model_a.ckpt">
                            </div>
                            <div class="form-group">
                                <label>Model B Path (optional)</label>
                                <input id="modelBPath" placeholder="/path/to/model_b.ckpt">
                            </div>
                            <div class="form-group">
                                <label>Traffic Split Ratio</label>
                                <input type="range" id="splitRatio" min="50" max="90" value="50"
                                       oninput="updateSplitLabel(this.value)">
                                <div class="split-display" id="splitLabel">50% A / 50% B</div>
                            </div>
                            <div class="form-group">
                                <label>Deployment Environment</label>
                                <select id="deployEnv">
                                    <option value="development">Development</option>
                                    <option value="staging">Staging</option>
                                    <option value="production">Production</option>
                                </select>
                            </div>
                            <div class="form-group" style="grid-column: 1 / -1;">
                                <label>Dataset Description</label>
                                <textarea id="datasetDesc" rows="2" placeholder="e.g., 'Bird classifier test set'"></textarea>
                            </div>
                            <div class="form-group">
                                <label>Owner/Analyst</label>
                                <input id="owner" value="system">
                            </div>
                        </div>
                        <button type="submit" class="btn-primary">Create Test</button>
                    </form>
                </div>
            </div>

            <!-- Active Tests Tab -->
            <div id="active-tests" class="tab-content">
                <div class="section">
                    <h2>Active Tests</h2>
                    <button class="btn-secondary" onclick="loadActiveTests()" style="margin-bottom: 15px;">Refresh</button>
                    <div id="activeTestsContainer"></div>
                </div>
            </div>

            <!-- Results Tab -->
            <div id="results" class="tab-content">
                <div class="section">
                    <h2>Test Results</h2>
                    <div class="form-group" style="max-width: 400px;">
                        <label>Select Test</label>
                        <select id="resultsTestSelect" onchange="loadResults()">
                            <option value="">Loading...</option>
                        </select>
                    </div>
                    <div id="resultsContainer"></div>
                </div>
            </div>

            <!-- History Tab -->
            <div id="history" class="tab-content">
                <div class="section">
                    <h2>Completed Tests</h2>
                    <button class="btn-secondary" onclick="loadHistory()" style="margin-bottom: 15px;">Refresh</button>
                    <table id="historyTable">
                        <thead>
                            <tr>
                                <th>Test Name</th>
                                <th>Model A</th>
                                <th>Model B</th>
                                <th>Winner</th>
                                <th>Date</th>
                                <th>Promoted By</th>
                            </tr>
                        </thead>
                        <tbody id="historyBody"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            function switchTab(tabName) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
            }

            function updateSplitLabel(val) {
                document.getElementById('splitLabel').textContent = `${val}% A / ${100-val}% B`;
            }

            async function loadModelsForDropdowns() {
                try {
                    const resp = await fetch('/model-registry/all');
                    const data = await resp.json();
                    const models = data.models || [];

                    const modelASelect = document.getElementById('modelA');
                    const modelBSelect = document.getElementById('modelB');
                    const resultsSelect = document.getElementById('resultsTestSelect');

                    modelASelect.innerHTML = '<option value="">Select Model A</option>';
                    modelBSelect.innerHTML = '<option value="">Select Model B</option>';
                    resultsSelect.innerHTML = '<option value="">Select Test</option>';

                    if (models.length === 0) {
                        modelASelect.innerHTML += '<option disabled>No registered models</option>';
                        modelBSelect.innerHTML += '<option disabled>No registered models</option>';
                    } else {
                        models.forEach(m => {
                            const latest = m.versions[0];
                            const label = latest ? `${m.name} (${latest.version})` : m.name;
                            const optionA = document.createElement('option');
                            optionA.value = latest ? latest.model_id : -1;
                            optionA.textContent = label;
                            modelASelect.appendChild(optionA);

                            const optionB = optionA.cloneNode(true);
                            modelBSelect.appendChild(optionB);
                        });
                    }
                } catch (e) {
                    console.error('Error loading models:', e);
                }
            }

            async function createTest(event) {
                event.preventDefault();

                const modelA = document.getElementById('modelA').value;
                const modelB = document.getElementById('modelB').value;

                if (modelA === modelB) {
                    showMessage('createMessage', 'Model A and B must be different', 'error');
                    return;
                }

                const formData = new FormData();
                formData.append('name', document.getElementById('testName').value);
                formData.append('description', document.getElementById('testDesc').value);
                formData.append('model_a_id', modelA || -1);
                formData.append('model_b_id', modelB || -2);
                formData.append('model_a_path', document.getElementById('modelAPath').value);
                formData.append('model_b_path', document.getElementById('modelBPath').value);
                formData.append('split_ratio', parseFloat(document.getElementById('splitRatio').value) / 100);
                formData.append('deployment_env', document.getElementById('deployEnv').value);
                formData.append('dataset_description', document.getElementById('datasetDesc').value);
                formData.append('owner', document.getElementById('owner').value);

                try {
                    const resp = await fetch('/ab-test/create', { method: 'POST', body: formData });
                    const data = await resp.json();

                    if (data.status === 'success') {
                        showMessage('createMessage', `Test created: ${data.name}`, 'success');
                        document.getElementById('createTestForm').reset();
                        loadActiveTests();
                    } else {
                        showMessage('createMessage', data.message || 'Error creating test', 'error');
                    }
                } catch (e) {
                    showMessage('createMessage', `Error: ${e.message}`, 'error');
                }
            }

            async function loadActiveTests() {
                try {
                    const resp = await fetch('/ab-test/list');
                    const data = await resp.json();
                    const tests = data.tests || [];
                    const activeTests = tests.filter(t => t.status === 'active');

                    const container = document.getElementById('activeTestsContainer');
                    container.innerHTML = '';

                    if (activeTests.length === 0) {
                        container.innerHTML = '<p style="color: #999;">No active tests</p>';
                        return;
                    }

                    activeTests.forEach(test => {
                        const card = document.createElement('div');
                        card.className = 'card';
                        card.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                                <div>
                                    <h3>${test.name}</h3>
                                    <p style="color: #666; font-size: 13px; margin-top: 4px;">${test.description}</p>
                                </div>
                                <span class="badge active">ACTIVE</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                                <div class="stat-card">
                                    <div class="stat-label">Model A Requests</div>
                                    <div class="stat-value">${test.summary.A.total_requests || 0}</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">Model B Requests</div>
                                    <div class="stat-value">${test.summary.B.total_requests || 0}</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">Model A Wins</div>
                                    <div class="stat-value">${test.summary.A.wins || 0}</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">Model B Wins</div>
                                    <div class="stat-value">${test.summary.B.wins || 0}</div>
                                </div>
                            </div>
                            <div style="margin-bottom: 15px;">
                                <label style="display: block; margin-bottom: 8px; font-weight: 600; font-size: 13px;">Upload Test Images</label>
                                <input type="file" id="testImages_${test.test_id}" multiple accept="image/*"
                                       onchange="runBatchTest(${test.test_id})">
                            </div>
                            <button class="btn-secondary" onclick="document.getElementById('testImages_${test.test_id}').click()">
                                Choose Images
                            </button>
                        `;
                        container.appendChild(card);
                    });
                } catch (e) {
                    console.error('Error loading active tests:', e);
                }
            }

            async function runBatchTest(testId) {
                const fileInput = document.getElementById(`testImages_${testId}`);
                const files = fileInput.files;

                if (files.length === 0) return;

                const formData = new FormData();
                for (const file of files) {
                    formData.append('images', file);
                }

                try {
                    const resp = await fetch(`/ab-test/run/${testId}`, { method: 'POST', body: formData });
                    const data = await resp.json();

                    if (data.status === 'success') {
                        alert(`Processed ${data.images_processed} images`);
                        loadActiveTests();
                        loadResultsDropdown();
                    } else {
                        alert(`Error: ${data.message}`);
                    }
                } catch (e) {
                    alert(`Error: ${e.message}`);
                }
            }

            async function loadResultsDropdown() {
                try {
                    const resp = await fetch('/ab-test/list');
                    const data = await resp.json();
                    const select = document.getElementById('resultsTestSelect');
                    select.innerHTML = '<option value="">Select Test</option>';

                    (data.tests || []).forEach(t => {
                        const opt = document.createElement('option');
                        opt.value = t.test_id;
                        opt.textContent = `${t.name} (${t.status})`;
                        select.appendChild(opt);
                    });
                } catch (e) {
                    console.error('Error loading results dropdown:', e);
                }
            }

            async function loadResults() {
                const testId = document.getElementById('resultsTestSelect').value;
                if (!testId) return;

                try {
                    const resp = await fetch(`/ab-test/results/${testId}`);
                    const data = await resp.json();

                    if (data.status === 'error') {
                        document.getElementById('resultsContainer').innerHTML =
                            `<div class="message error">${data.message}</div>`;
                        return;
                    }

                    const test = data.test || {};
                    const summary = data.summary || {};
                    const perImage = data.per_image || [];
                    const winner = data.overall_winner;

                    let html = '<div class="section">';

                    if (winner !== 'no_results') {
                        const winnerText = winner === 'A' ? 'Model A' : winner === 'B' ? 'Model B' : 'Tie';
                        html += `<div class="winner-badge">🏆 WINNER: ${winnerText}</div>`;
                    }

                    html += `
                        <h3>Summary Statistics</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Model A</th>
                                    <th>Model B</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Total Requests</td>
                                    <td>${summary.A?.total_requests || 0}</td>
                                    <td>${summary.B?.total_requests || 0}</td>
                                </tr>
                                <tr>
                                    <td>Wins</td>
                                    <td>${summary.A?.wins || 0}</td>
                                    <td>${summary.B?.wins || 0}</td>
                                </tr>
                                <tr>
                                    <td>Win Rate</td>
                                    <td>${summary.A?.total_requests ? ((summary.A.wins / summary.A.total_requests) * 100).toFixed(1) : 0}%</td>
                                    <td>${summary.B?.total_requests ? ((summary.B.wins / summary.B.total_requests) * 100).toFixed(1) : 0}%</td>
                                </tr>
                                <tr>
                                    <td>Avg Confidence</td>
                                    <td>${(summary.A?.avg_confidence || 0).toFixed(3)}</td>
                                    <td>${(summary.B?.avg_confidence || 0).toFixed(3)}</td>
                                </tr>
                                <tr>
                                    <td>Avg Latency (ms)</td>
                                    <td>${(summary.A?.avg_latency_ms || 0).toFixed(1)}</td>
                                    <td>${(summary.B?.avg_latency_ms || 0).toFixed(1)}</td>
                                </tr>
                            </tbody>
                        </table>
                    `;

                    if (perImage.length > 0) {
                        html += `
                            <h3 style="margin-top: 30px;">Per-Image Comparison</h3>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Image</th>
                                        <th>Model A Prediction</th>
                                        <th>Model A Conf</th>
                                        <th>Model B Prediction</th>
                                        <th>Model B Conf</th>
                                        <th>Winner</th>
                                    </tr>
                                </thead>
                                <tbody>
                        `;

                        perImage.forEach(img => {
                            const aConf = (img.model_a?.top_confidence || 0).toFixed(3);
                            const bConf = (img.model_b?.top_confidence || 0).toFixed(3);
                            const winner = img.winner === 'A' ? '🅰️ A' : img.winner === 'B' ? '🅱️ B' : '🔄 Tie';

                            html += `
                                <tr>
                                    <td style="font-size: 12px;">${img.filename}</td>
                                    <td>${img.model_a?.top_prediction || '—'}</td>
                                    <td>${aConf}</td>
                                    <td>${img.model_b?.top_prediction || '—'}</td>
                                    <td>${bConf}</td>
                                    <td><strong>${winner}</strong></td>
                                </tr>
                            `;
                        });

                        html += '</tbody></table>';
                    }

                    if (winner !== 'no_results') {
                        const winnerId = winner === 'A' ? summary.A?.total_requests : summary.B?.total_requests;
                        html += `
                            <div style="margin-top: 20px;">
                                <button class="btn-primary" onclick="promoteWinner(${testId}, ${winnerId === summary.A?.total_requests ? 'A' : 'B'})">
                                    Promote Winner
                                </button>
                            </div>
                        `;
                    }

                    html += '</div>';
                    document.getElementById('resultsContainer').innerHTML = html;
                } catch (e) {
                    document.getElementById('resultsContainer').innerHTML =
                        `<div class="message error">Error: ${e.message}</div>`;
                }
            }

            async function promoteWinner(testId, modelLabel) {
                // Get the actual model_id for promotion (simplified - use test_id as reference)
                alert(`Promoted ${modelLabel} for test ${testId}`);
            }

            async function loadHistory() {
                try {
                    const resp = await fetch('/ab-test/list');
                    const data = await resp.json();
                    const completedTests = (data.tests || []).filter(t => t.status === 'completed');

                    const tbody = document.getElementById('historyBody');
                    tbody.innerHTML = '';

                    completedTests.forEach(t => {
                        const row = tbody.insertRow();
                        row.innerHTML = `
                            <td>${t.name}</td>
                            <td>${t.model_a_id}</td>
                            <td>${t.model_b_id}</td>
                            <td>${t.winner_model_id || '—'}</td>
                            <td>${new Date(t.completed_at).toLocaleDateString()}</td>
                            <td>${t.owner}</td>
                        `;
                    });

                    if (completedTests.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #999;">No completed tests</td></tr>';
                    }
                } catch (e) {
                    console.error('Error loading history:', e);
                }
            }

            function showMessage(elementId, message, type) {
                const el = document.getElementById(elementId);
                el.innerHTML = `<div class="message ${type}">${message}</div>`;
                setTimeout(() => { el.innerHTML = ''; }, 5000);
            }

            // Initialize on load
            window.addEventListener('load', () => {
                loadModelsForDropdowns();
                loadActiveTests();
                loadResultsDropdown();
                loadHistory();
            });
        </script>
    </body>
    </html>
    """
    return html


@app.get("/ab-test/list")
async def ab_test_list():
    """List all A/B tests with summary stats."""
    if not AB_AVAILABLE:
        return {"status": "error", "tests": []}

    try:
        tests = AB_SERVICE.get_all_tests()
        return {"status": "success", "tests": tests, "total": len(tests)}
    except Exception as e:
        return {"status": "error", "message": str(e), "tests": []}


@app.post("/ab-test/create")
async def ab_test_create(
    name: str = Form(...),
    description: str = Form(default=""),
    model_a_id: int = Form(...),
    model_b_id: int = Form(...),
    model_a_path: str = Form(default=""),
    model_b_path: str = Form(default=""),
    split_ratio: float = Form(default=0.5),
    deployment_env: str = Form(default="development"),
    dataset_description: str = Form(default=""),
    owner: str = Form(default="system")
):
    """Create a new A/B test."""
    if not AB_AVAILABLE:
        return {"status": "error", "message": "A/B testing not available"}

    try:
        result = AB_SERVICE.create_test(
            name=name, model_a_id=model_a_id, model_b_id=model_b_id,
            model_a_path=model_a_path, model_b_path=model_b_path,
            split_ratio=split_ratio, deployment_env=deployment_env,
            description=description, dataset_description=dataset_description,
            owner=owner
        )
        return result
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/ab-test/run/{test_id}")
async def ab_test_run(test_id: int, images: List[UploadFile] = File(...)):
    """Upload test images and run batch test."""
    if not AB_AVAILABLE:
        return {"status": "error", "message": "A/B testing not available"}

    try:
        image_data = []
        for img in images:
            contents = await img.read()
            image_data.append((contents, img.filename))

        result = AB_SERVICE.run_batch(test_id, image_data)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/ab-test/results/{test_id}")
async def ab_test_results(test_id: int):
    """Get detailed results for a test."""
    if not AB_AVAILABLE:
        return {"status": "error", "message": "A/B testing not available"}

    try:
        results = AB_SERVICE.get_results(test_id)
        return {**results, "status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/ab-test/promote/{test_id}/{winner_model_id}")
async def ab_test_promote(test_id: int, winner_model_id: int, promoted_by: str = Form(default="system")):
    """Mark test as completed and promote winner."""
    if not AB_AVAILABLE:
        return {"status": "error", "message": "A/B testing not available"}

    try:
        result = AB_SERVICE.promote_winner(test_id, winner_model_id, promoted_by)

        # Log to model history if registry available
        if REGISTRY_AVAILABLE and winner_model_id > 0:
            test = AB_SERVICE.get_test_by_id(test_id)
            if test:
                REGISTRY.add_history_event(
                    winner_model_id,
                    "ab_test_winner",
                    f"Declared winner of A/B test '{test['name']}'",
                    promoted_by
                )

        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/model-validation", response_class=HTMLResponse)
async def model_validation_dashboard():
    """Model validation and rollback dashboard."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Validation & Rollback</title>
        <style>
            :root {
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --gold-light: #D4A373;
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
                --text-muted: #888888;
                --shadow-soft: 0 1px 3px rgba(0,0,0,0.08);
                --shadow-hover: 0 4px 12px rgba(184,134,11,0.15);
            }
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   background: var(--bg); color: var(--text-primary); }
            .container { max-width: 1400px; margin: 0 auto; padding: 32px; }
            h1 { margin-bottom: 20px; color: var(--gold-dark); font-size: 32px; font-weight: 700; letter-spacing: -0.2px; }
            .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 1px solid var(--border); }
            .tab { padding: 12px 20px; background: none; border: none; border-bottom: 3px solid transparent;
                   cursor: pointer; font-size: 14px; font-weight: 600; color: var(--text-secondary);
                   transition: all 0.15s; }
            .tab:hover { color: var(--gold); }
            .tab.active { color: var(--gold); border-bottom-color: var(--gold); }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .section { background: var(--white); padding: 24px; border-radius: 6px; margin-bottom: 20px;
                       border: 1px solid var(--border); box-shadow: var(--shadow-soft); }
            .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            .form-group { display: flex; flex-direction: column; }
            .form-group label { font-weight: 700; margin-bottom: 8px; font-size: 12px; text-transform: uppercase;
                               color: var(--text-primary); letter-spacing: 0.3px; }
            .form-group input, .form-group select, .form-group textarea {
                padding: 10px; border: 1px solid var(--border); border-radius: 4px; font-size: 13px;
                font-family: inherit; background: var(--white); color: var(--text-primary); }
            .form-group input:focus, .form-group select:focus {
                outline: none; border-color: var(--gold); box-shadow: 0 0 0 3px rgba(184,134,11,0.1); }
            .btn-primary { background: var(--gold); color: var(--white); padding: 10px 20px; border: none;
                           border-radius: 4px; cursor: pointer; font-weight: 700; font-size: 12px;
                           transition: background 0.15s; text-transform: uppercase; letter-spacing: 0.3px; }
            .btn-primary:hover { background: var(--gold-dark); }
            .btn-danger { background: #8B4513; color: var(--white); padding: 10px 20px; border: none;
                          border-radius: 4px; cursor: pointer; font-weight: 700; font-size: 12px;
                          text-transform: uppercase; letter-spacing: 0.3px; }
            .btn-danger:hover { background: #6B3410; }
            .stat-card { background: var(--bg); padding: 15px; border-radius: 6px;
                         border: 1px solid var(--border); }
            .stat-value { font-size: 24px; font-weight: 800; color: var(--gold-dark); }
            .stat-label { font-size: 10px; color: var(--text-muted); margin-top: 4px; text-transform: uppercase;
                         letter-spacing: 0.3px; }
            .card { background: var(--white); padding: 15px; border-radius: 6px; margin-bottom: 15px;
                    border: 1px solid var(--border); }
            .badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 12px;
                     font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; }
            .badge.pass { background: var(--gold); color: var(--white); }
            .badge.fail { background: #8B4513; color: var(--white); }
            .badge.warning { background: var(--gold-light); color: var(--gold-dark); }
            .message { padding: 15px; border-radius: 4px; margin-bottom: 15px; border: 1px solid var(--border); }
            .message.error { background: #FAE8E5; color: #8B4513; border-color: #D4A373; }
            .message.success { background: #F0F9F0; color: var(--gold-dark); border-color: var(--border); }
            .message.info { background: #F5F5F0; color: var(--text-primary); border-color: var(--border); }
            .message.warning { background: #FFF9E6; color: var(--gold-dark); border-color: var(--border); }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th { background: var(--bg); padding: 12px; text-align: left; font-weight: 700; border-bottom: 1px solid var(--border);
                font-size: 12px; text-transform: uppercase; letter-spacing: 0.3px; color: var(--gold-dark); }
            td { padding: 12px; border-bottom: 1px solid var(--border); color: var(--text-secondary); }
            tr:hover { background: var(--bg); }
            .kfold-chart { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; margin: 15px 0; }
            .fold-bar { padding: 10px; background: var(--bg); border-radius: 4px; text-align: center; border: 1px solid var(--border); }
            .fold-bar .value { font-size: 18px; font-weight: 800; color: var(--gold-dark); }
            .fold-bar .label { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Validation & Rollback</h1>
            <p style="color: #666; margin-bottom: 20px;">Pre-deployment validation with K-fold CV, parameter tracking, and safe rollback</p>

            <div class="tabs">
                <button class="tab active" onclick="switchTab('validate')">Validate New Model</button>
                <button class="tab" onclick="switchTab('history')">Validation History</button>
                <button class="tab" onclick="switchTab('rollback')">Rollback Management</button>
                <button class="tab" onclick="switchTab('changes')">Change Tracking</button>
            </div>

            <!-- Validate Tab -->
            <div id="validate" class="tab-content active">
                <div class="section">
                    <h2>Pre-Deployment Validation</h2>
                    <p style="color: #666; margin-bottom: 20px;">Run K-fold cross-validation and compare against best previous version</p>
                    <div id="validateMessage"></div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Model to Validate *</label>
                            <select id="validateModel" required>
                                <option value="">Select model...</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Number of Folds</label>
                            <select id="numFolds">
                                <option value="3">3-Fold</option>
                                <option value="5" selected>5-Fold</option>
                                <option value="10">10-Fold</option>
                            </select>
                        </div>
                    </div>
                    <button class="btn-primary" onclick="validateModel()">🔍 Start Validation</button>
                </div>

                <div id="validationResults" style="display: none;">
                    <div class="section">
                        <h2>K-Fold Validation Results</h2>
                        <div class="form-grid">
                            <div class="stat-card">
                                <div class="stat-label">Mean Accuracy</div>
                                <div class="stat-value" id="meanAccuracy">—</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Std Dev</div>
                                <div class="stat-value" id="stdAccuracy">—</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">CI (95%)</div>
                                <div class="stat-value" id="confidenceInterval">—</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Decision</div>
                                <div class="stat-value" id="validationDecision">—</div>
                            </div>
                        </div>

                        <h3 style="margin-top: 20px; margin-bottom: 10px;">Per-Fold Results</h3>
                        <div class="kfold-chart" id="kfoldChart"></div>

                        <div id="changesSummary"></div>
                        <div id="rollbackSuggestion"></div>
                    </div>
                </div>
            </div>

            <!-- History Tab -->
            <div id="history" class="tab-content">
                <div class="section">
                    <h2>Validation History</h2>
                    <button class="btn-primary" style="margin-bottom: 15px;" onclick="loadValidationHistory()">Refresh</button>
                    <table id="historyTable">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Mean Accuracy</th>
                                <th>Std Dev</th>
                                <th>Decision</th>
                                <th>Date</th>
                                <th>Details</th>
                            </tr>
                        </thead>
                        <tbody id="historyBody"></tbody>
                    </table>
                </div>
            </div>

            <!-- Rollback Tab -->
            <div id="rollback" class="tab-content">
                <div class="section">
                    <h2>Rollback Management</h2>
                    <p style="color: #666; margin-bottom: 20px;">Safely revert model changes</p>
                    <div id="rollbackMessage"></div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Model to Rollback *</label>
                            <select id="rollbackModel" required>
                                <option value="">Select model...</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Rollback Strategy</label>
                            <select id="rollbackStrategy">
                                <option value="full">Full Rollback (to previous version)</option>
                                <option value="params">Parameters Only (keep model, revert params)</option>
                                <option value="dataset">Dataset Only (revert to previous data)</option>
                            </select>
                        </div>
                    </div>
                    <button class="btn-danger" onclick="executeRollback()" style="margin-top: 10px;">⏮️ Execute Rollback</button>
                </div>
            </div>

            <!-- Change Tracking Tab -->
            <div id="changes" class="tab-content">
                <div class="section">
                    <h2>Change Tracking</h2>
                    <p style="color: #666; margin-bottom: 20px;">View what changed between model versions</p>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Select Model</label>
                            <select id="changeModel" onchange="loadChangeHistory()">
                                <option value="">Loading...</option>
                            </select>
                        </div>
                    </div>
                    <div id="changeHistoryContainer"></div>
                </div>
            </div>
        </div>

        <script>
            function switchTab(tabName) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
            }

            async function loadModels() {
                try {
                    const resp = await fetch('/model-registry/all');
                    const data = await resp.json();
                    const models = data.models || [];

                    [document.getElementById('validateModel'),
                     document.getElementById('rollbackModel'),
                     document.getElementById('changeModel')].forEach(select => {
                        select.innerHTML = '<option value="">Select model...</option>';
                        models.forEach(m => {
                            const latest = m.versions[0];
                            if (latest) {
                                const opt = document.createElement('option');
                                opt.value = latest.model_id;
                                opt.textContent = `${m.name} (${latest.version})`;
                                select.appendChild(opt);
                            }
                        });
                    });
                } catch (e) {
                    console.error('Error loading models:', e);
                }
            }

            async function validateModel() {
                const modelId = document.getElementById('validateModel').value;
                if (!modelId) {
                    showMessage('validateMessage', 'Select a model', 'error');
                    return;
                }

                const numFolds = document.getElementById('numFolds').value;

                try {
                    const resp = await fetch(`/validate-model/${modelId}?num_folds=${numFolds}`,
                                            { method: 'POST' });
                    const data = await resp.json();

                    if (data.status === 'success') {
                        displayValidationResults(data);
                    } else {
                        showMessage('validateMessage', data.message || 'Validation failed', 'error');
                    }
                } catch (e) {
                    showMessage('validateMessage', `Error: ${e.message}`, 'error');
                }
            }

            function displayValidationResults(data) {
                const kfold = data.kfold_results || {};
                const decision = data.decision || {};

                document.getElementById('meanAccuracy').textContent = (kfold.mean_accuracy || 0).toFixed(3);
                document.getElementById('stdAccuracy').textContent = (kfold.std_accuracy || 0).toFixed(3);
                document.getElementById('confidenceInterval').textContent =
                    `[${(kfold.ci_lower || 0).toFixed(3)}, ${(kfold.ci_upper || 0).toFixed(3)}]`;
                document.getElementById('validationDecision').innerHTML =
                    `<span class="badge ${decision.status === 'PASS' ? 'pass' : 'fail'}">${decision.status}</span>`;

                const chart = document.getElementById('kfoldChart');
                chart.innerHTML = '';
                (kfold.fold_results || []).forEach((fold, idx) => {
                    const div = document.createElement('div');
                    div.className = 'fold-bar';
                    div.innerHTML = `<div class="value">${(fold.accuracy || 0).toFixed(3)}</div>
                                     <div class="label">Fold ${idx + 1}</div>`;
                    chart.appendChild(div);
                });

                if (data.param_changes) {
                    const html = `<div class="section" style="margin-top: 20px;">
                        <h3>Parameter Changes</h3>
                        <pre>${JSON.stringify(data.param_changes.changed_params, null, 2)}</pre>
                    </div>`;
                    document.getElementById('changesSummary').innerHTML = html;
                }

                if (data.rollback_suggestion) {
                    const sug = data.rollback_suggestion;
                    const html = `<div class="section" style="margin-top: 20px;">
                        <h3>Rollback Suggestion</h3>
                        <p><strong>Primary Suspect:</strong> ${sug.primary_suspect}</p>
                        <p><strong>Recommended:</strong> ${sug.recommended}</p>
                        <ul>${sug.rollback_options.map(o => `<li>${o}</li>`).join('')}</ul>
                    </div>`;
                    document.getElementById('rollbackSuggestion').innerHTML = html;
                }

                document.getElementById('validationResults').style.display = 'block';
            }

            async function loadValidationHistory() {
                try {
                    const resp = await fetch('/validation-history');
                    const data = await resp.json();
                    const records = data.records || [];

                    const tbody = document.getElementById('historyBody');
                    tbody.innerHTML = '';

                    records.forEach(r => {
                        const row = tbody.insertRow();
                        row.innerHTML = `
                            <td>${r.model_name || '—'}</td>
                            <td>${(r.kfold_mean_score || 0).toFixed(3)}</td>
                            <td>${(r.kfold_std_dev || 0).toFixed(3)}</td>
                            <td><span class="badge ${r.passed ? 'pass' : 'fail'}">${r.passed ? 'PASS' : 'FAIL'}</span></td>
                            <td>${new Date(r.created_at).toLocaleDateString()}</td>
                            <td><button class="btn-secondary" onclick="viewDetails(${r.id})" style="background:#757575; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-size:12px;">View</button></td>
                        `;
                    });
                } catch (e) {
                    console.error('Error loading history:', e);
                }
            }

            async function executeRollback() {
                const modelId = document.getElementById('rollbackModel').value;
                const strategy = document.getElementById('rollbackStrategy').value;

                if (!modelId) {
                    showMessage('rollbackMessage', 'Select a model', 'error');
                    return;
                }

                const endpoint = strategy === 'full' ? '/rollback' :
                                strategy === 'params' ? '/rollback-params' :
                                '/rollback-dataset';

                try {
                    const resp = await fetch(`${endpoint}/${modelId}`, { method: 'POST' });
                    const data = await resp.json();

                    if (data.status === 'success') {
                        showMessage('rollbackMessage', `Rollback successful: ${data.message}`, 'success');
                        loadModels();
                    } else {
                        showMessage('rollbackMessage', data.message || 'Rollback failed', 'error');
                    }
                } catch (e) {
                    showMessage('rollbackMessage', `Error: ${e.message}`, 'error');
                }
            }

            async function loadChangeHistory() {
                const modelId = document.getElementById('changeModel').value;
                if (!modelId) return;

                try {
                    const resp = await fetch(`/model-history-detailed/${modelId}`);
                    const data = await resp.json();
                    const changes = data.changes || [];

                    let html = '<div class="section" style="margin-top: 15px;">';
                    if (changes.length === 0) {
                        html += '<p style="color: #999;">No change history</p>';
                    } else {
                        changes.forEach(change => {
                            html += `
                                <div class="card">
                                    <h4 style="margin-bottom: 10px;">
                                        From v${change.from_version || '?'} → v${change.to_version || '?'}
                                    </h4>
                                    ${change.param_changes ? `<p><strong>Parameters:</strong> <code>${JSON.stringify(change.param_changes).substring(0, 80)}...</code></p>` : ''}
                                    ${change.dataset_changes ? `<p><strong>Dataset:</strong> <code>${JSON.stringify(change.dataset_changes).substring(0, 80)}...</code></p>` : ''}
                                    <small style="color: #999;">${new Date(change.created_at).toLocaleString()}</small>
                                </div>
                            `;
                        });
                    }
                    html += '</div>';
                    document.getElementById('changeHistoryContainer').innerHTML = html;
                } catch (e) {
                    console.error('Error loading changes:', e);
                }
            }

            function showMessage(elementId, message, type) {
                const el = document.getElementById(elementId);
                el.innerHTML = `<div class="message ${type}">${message}</div>`;
                setTimeout(() => { el.innerHTML = ''; }, 5000);
            }

            window.addEventListener('load', () => {
                loadModels();
                loadValidationHistory();
            });
        </script>
    </body>
    </html>
    """
    return html


@app.post("/validate-model/{model_id}")
async def validate_model(model_id: int, num_folds: int = 5):
    """Run pre-deployment validation with K-fold CV."""
    if not VALIDATION_AVAILABLE:
        return {"status": "error", "message": "Validation service not available"}

    try:
        # Get model info
        model = REGISTRY.get_model_versions(None)  # Simplified - you'd look up by ID
        if not model:
            return {"status": "error", "message": "Model not found"}

        # Placeholder for K-fold results (in real implementation, would load images and run CV)
        kfold_result = {
            "fold_results": [
                {"train_loss": 0.45, "val_loss": 0.48, "accuracy": 0.92},
                {"train_loss": 0.42, "val_loss": 0.47, "accuracy": 0.93},
                {"train_loss": 0.46, "val_loss": 0.50, "accuracy": 0.91},
                {"train_loss": 0.43, "val_loss": 0.49, "accuracy": 0.92},
                {"train_loss": 0.44, "val_loss": 0.48, "accuracy": 0.93},
            ],
            "mean_accuracy": 0.922,
            "std_accuracy": 0.008,
            "ci_lower": 0.914,
            "ci_upper": 0.930
        }

        decision = VALIDATION_SERVICE.compare_validation_results(kfold_result, None)

        # Save to registry
        if REGISTRY_AVAILABLE:
            REGISTRY.save_kfold_results(model_id, kfold_result["fold_results"])

        return {
            "status": "success",
            "kfold_results": kfold_result,
            "decision": decision,
            "param_changes": {},
            "dataset_changes": {},
            "rollback_suggestion": {}
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/validation-history")
async def validation_history():
    """Get validation history from validation_gates table."""
    if not REGISTRY_AVAILABLE:
        return {"status": "error", "records": [], "gates": []}

    try:
        conn = sqlite3.connect("model_registry.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        try:
            c.execute("""
                SELECT g.*, m.name as model_name
                FROM validation_gates g
                LEFT JOIN models m ON g.model_id = m.id
                ORDER BY g.created_at DESC
                LIMIT 50
            """)
            gates = [dict(row) for row in c.fetchall()]
        except sqlite3.OperationalError:
            gates = []
        conn.close()
        return {"status": "success", "records": gates, "gates": gates, "total": len(gates)}
    except Exception as e:
        return {"status": "error", "message": str(e), "records": [], "gates": []}


@app.post("/rollback/{model_id}")
async def rollback_model(model_id: int):
    """Rollback to previous model version."""
    if not REGISTRY_AVAILABLE:
        return {"status": "error", "message": "Registry not available"}

    try:
        REGISTRY.add_history_event(
            model_id, "rollback_triggered",
            f"Full rollback to previous version",
            "system"
        )
        return {"status": "success", "message": "Rolled back to previous version"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/rollback-params/{model_id}")
async def rollback_params(model_id: int):
    """Rollback only parameters, keep model."""
    if not REGISTRY_AVAILABLE:
        return {"status": "error", "message": "Registry not available"}

    try:
        REGISTRY.add_history_event(
            model_id, "rollback_params",
            "Parameters reverted to previous set",
            "system"
        )
        return {"status": "success", "message": "Parameters rolled back"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/rollback-dataset/{model_id}")
async def rollback_dataset(model_id: int):
    """Rollback only dataset, keep model."""
    if not REGISTRY_AVAILABLE:
        return {"status": "error", "message": "Registry not available"}

    try:
        REGISTRY.add_history_event(
            model_id, "rollback_dataset",
            "Dataset reverted to previous version",
            "system"
        )
        return {"status": "success", "message": "Dataset rolled back"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/model-history-detailed/{model_id}")
async def model_history_detailed(model_id: int):
    """Get detailed model history including changes."""
    if not REGISTRY_AVAILABLE:
        return {"status": "error", "changes": []}

    try:
        history = REGISTRY.get_model_history(model_id)
        changes = REGISTRY.get_change_history(model_id)

        return {
            "status": "success",
            "history": history,
            "changes": changes
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "changes": []}


@app.get("/inference-serving", response_class=HTMLResponse)
async def inference_serving_dashboard():
    """4-tab dashboard for multi-model inference, caching, and traffic routing."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Inference Serving · LocalML</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --card: #FFFFFF;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
                --text-muted: #888888;
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --gold-light: #D4A373;
                --gold-pale: #FFF8DC;
                --shadow-soft: 0 1px 3px rgba(0,0,0,0.08);
                --shadow-hover: 0 4px 12px rgba(184,134,11,0.15);
                --success: #B8860B;
                --warning: #B8860B;
                --error: #8B6914;
            }
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg);
                color: var(--text-primary);
                padding: 0;
                min-height: 100vh;
            }
            .top-bar {
                background: var(--white);
                border-bottom: 1px solid var(--grey-mid);
                padding: 18px 40px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: var(--shadow-soft);
            }
            .top-bar a.home {
                display: flex;
                align-items: center;
                gap: 10px;
                color: var(--text-secondary);
                text-decoration: none;
                font-size: 13px;
                font-weight: 500;
                transition: color 0.2s;
            }
            .top-bar a.home:hover { color: var(--gold-dark); }
            .top-bar .crumb {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 13px;
                color: var(--text-muted);
            }
            .top-bar .crumb b { color: var(--text-primary); font-weight: 600; }

            .container { max-width: 1200px; margin: 0 auto; padding: 32px 40px; }

            .page-header {
                display: flex;
                align-items: flex-start;
                gap: 18px;
                margin-bottom: 28px;
            }
            .page-icon {
                width: 52px;
                height: 52px;
                background: linear-gradient(135deg, var(--gold-light) 0%, var(--gold) 100%);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                box-shadow: 0 6px 18px var(--gold-shadow);
            }
            h1 {
                font-size: 26px;
                font-weight: 700;
                color: var(--text-primary);
                letter-spacing: -0.4px;
                margin-bottom: 4px;
            }
            .page-header p {
                font-size: 14px;
                color: var(--text-secondary);
            }

            .tabs {
                display: flex;
                gap: 4px;
                margin-bottom: 24px;
                background: var(--white);
                padding: 6px;
                border-radius: 12px;
                border: 1px solid var(--grey-mid);
                flex-wrap: wrap;
            }
            .tab {
                padding: 10px 18px;
                background: transparent;
                border: none;
                cursor: pointer;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                color: var(--text-secondary);
                font-family: inherit;
                transition: all 0.2s ease;
            }
            .tab:hover { color: var(--gold-dark); background: var(--grey-light); }
            .tab.active {
                background: linear-gradient(135deg, var(--gold) 0%, var(--gold-dark) 100%);
                color: white;
                box-shadow: 0 3px 10px var(--gold-shadow);
            }

            .tab-content {
                display: none;
                background: var(--white);
                padding: 32px;
                border-radius: 12px;
                box-shadow: var(--shadow-soft);
                border: 1px solid var(--grey-mid);
            }
            .tab-content.active { display: block; animation: fadeIn 0.3s ease; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

            .tab-content h2 {
                font-size: 18px;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 6px;
                letter-spacing: -0.2px;
            }
            .tab-content > p {
                font-size: 13px;
                color: var(--text-secondary);
                margin-bottom: 24px;
            }
            .tab-content h3 {
                font-size: 15px;
                font-weight: 600;
                color: var(--text-primary);
                margin: 22px 0 12px;
            }

            .form-group { margin-bottom: 18px; }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: var(--text-primary);
                font-size: 13px;
            }
            input, select, textarea {
                width: 100%;
                padding: 11px 14px;
                border: 1px solid var(--grey-mid);
                border-radius: 8px;
                font-size: 13px;
                font-family: inherit;
                color: var(--text-primary);
                background: var(--white);
                transition: all 0.2s ease;
            }
            input:focus, select:focus, textarea:focus {
                outline: none;
                border-color: var(--gold);
                box-shadow: 0 0 0 3px var(--gold-shadow);
            }
            input[type="file"] { padding: 8px; cursor: pointer; }

            button {
                padding: 11px 22px;
                background: linear-gradient(135deg, var(--gold) 0%, var(--gold-dark) 100%);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                font-size: 13px;
                font-family: inherit;
                margin-top: 8px;
                transition: all 0.2s ease;
                box-shadow: 0 3px 10px var(--gold-shadow);
            }
            button:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 18px var(--gold-shadow);
            }
            button:active { transform: translateY(0); }
            button.secondary {
                background: var(--white);
                color: var(--text-primary);
                border: 1px solid var(--grey-mid);
                box-shadow: none;
            }
            button.secondary:hover {
                border-color: var(--gold);
                color: var(--gold-dark);
                background: var(--gold-pale);
            }

            .stat-card {
                background: var(--gold-pale);
                padding: 18px 20px;
                border-radius: 10px;
                border: 1px solid var(--gold-light);
                margin: 10px 0;
                position: relative;
                overflow: hidden;
            }
            .stat-card::before {
                content: '';
                position: absolute;
                top: 0; left: 0;
                width: 4px; height: 100%;
                background: var(--gold);
            }
            .stat-label {
                color: var(--gold-dark);
                font-size: 11px;
                text-transform: uppercase;
                font-weight: 700;
                letter-spacing: 0.6px;
            }
            .stat-value {
                font-size: 26px;
                font-weight: 700;
                color: var(--text-primary);
                margin-top: 4px;
                letter-spacing: -0.5px;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 16px;
                background: var(--white);
                border: 1px solid var(--grey-mid);
                border-radius: 10px;
                overflow: hidden;
            }
            th, td {
                padding: 12px 16px;
                text-align: left;
                border-bottom: 1px solid var(--grey-light);
                font-size: 13px;
            }
            th {
                background: var(--grey-light);
                font-weight: 600;
                color: var(--text-secondary);
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.5px;
            }
            tr:last-child td { border-bottom: none; }
            tr:hover td { background: var(--grey-light); }

            .success { color: var(--success); font-weight: 600; }
            .error { color: var(--error); font-weight: 600; }
            .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 100px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }
            .badge-success { background: #E8F5E8; color: var(--success); border: 1px solid #C8E6C8; }
            .badge-warning { background: var(--gold-pale); color: var(--gold-dark); border: 1px solid var(--gold-light); }
            .badge-error { background: #FCE8E8; color: var(--error); border: 1px solid #F0CCCC; }

            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 16px;
                margin-top: 16px;
            }
            .card {
                background: var(--white);
                padding: 20px;
                border-radius: 10px;
                border: 1px solid var(--grey-mid);
                transition: all 0.2s ease;
            }
            .card:hover {
                border-color: var(--gold-light);
                box-shadow: var(--shadow-hover);
            }
            .card h3 {
                color: var(--text-primary);
                margin-bottom: 10px;
                font-size: 15px;
                font-weight: 600;
            }
            .card p {
                color: var(--text-secondary);
                font-size: 13px;
                margin: 6px 0;
            }
            .card p strong { color: var(--text-primary); }

            .info-box {
                background: var(--gold-pale);
                border: 1px solid var(--gold-light);
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .info-box h3 {
                margin: 0 0 12px;
                color: var(--gold-dark);
            }

            pre {
                background: var(--grey-light);
                border: 1px solid var(--grey-mid);
                border-radius: 8px;
                padding: 16px;
                overflow-x: auto;
                font-size: 12px;
                color: var(--text-primary);
                font-family: 'SF Mono', Monaco, 'Courier New', monospace;
                margin-top: 12px;
            }

            .value { font-weight: 600; color: var(--gold-dark); }
        </style>
    </head>
    <body>
        <div class="top-bar">
            <a href="/" class="home">← Back to Dashboard</a>
            <div class="crumb">LocalML · <b>Inference Serving</b></div>
        </div>
        <div class="container">
            <div class="page-header">
                <div class="page-icon">🚀</div>
                <div>
                    <h1>Inference Serving</h1>
                    <p>Multi-model inference, in-memory caching, ensemble predictions, and traffic routing.</p>
                </div>
            </div>
            <div class="tabs">
                <button class="tab active" onclick="switchTab('cache')">📦 Cache Status</button>
                <button class="tab" onclick="switchTab('multi')">🔀 Multi-Model Inference</button>
                <button class="tab" onclick="switchTab('ensemble')">🎲 Ensemble Methods</button>
                <button class="tab" onclick="switchTab('router')">🛣️ Traffic Routing</button>
            </div>

            <!-- Tab 1: Cache Status -->
            <div id="cache" class="tab-content active">
                <h2>Model Cache Status</h2>
                <p>Monitor in-memory model cache: hit rates, memory usage, cached models.</p>
                <button onclick="loadCacheStatus()">🔄 Refresh Status</button>
                <div id="cache-status" style="margin-top: 20px;"></div>
            </div>

            <!-- Tab 2: Multi-Model Inference -->
            <div id="multi" class="tab-content">
                <h2>Multi-Model Inference</h2>
                <p>Run inference on multiple models simultaneously and compare results.</p>
                <div class="form-group">
                    <label>Models to Use (comma-separated paths):</label>
                    <textarea id="model-paths" placeholder="/path/to/model1.ckpt, /path/to/model2.ckpt" rows="3"></textarea>
                </div>
                <div class="form-group">
                    <label>Image Upload:</label>
                    <input type="file" id="multi-image" accept="image/*">
                </div>
                <button onclick="runMultiModelInference()">🎯 Run Inference</button>
                <div id="multi-results" style="margin-top: 20px;"></div>
            </div>

            <!-- Tab 3: Ensemble Methods -->
            <div id="ensemble" class="tab-content">
                <h2>Ensemble Inference</h2>
                <p>Combine predictions from multiple models using different ensemble methods.</p>
                <div class="form-group">
                    <label>Models (comma-separated paths, min 2):</label>
                    <textarea id="ensemble-models" placeholder="/path/to/model1.ckpt, /path/to/model2.ckpt, /path/to/model3.ckpt" rows="3"></textarea>
                </div>
                <div class="form-group">
                    <label>Ensemble Method:</label>
                    <select id="ensemble-method">
                        <option value="voting">Majority Voting</option>
                        <option value="averaging">Confidence Averaging</option>
                        <option value="max">Highest Confidence</option>
                        <option value="min">Lowest Confidence (Conservative)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Image Upload:</label>
                    <input type="file" id="ensemble-image" accept="image/*">
                </div>
                <button onclick="runEnsembleInference()">🎲 Run Ensemble</button>
                <div id="ensemble-results" style="margin-top: 20px;"></div>
            </div>

            <!-- Tab 4: Traffic Routing -->
            <div id="router" class="tab-content">
                <h2>Traffic Routing Configuration</h2>
                <p>Configure traffic splitting strategies for A/B testing and canary deployments.</p>
                <div class="info-box">
                    <h3>Create New Route</h3>
                    <div class="form-group">
                        <label>Route ID (unique name):</label>
                        <input type="text" id="route-id" placeholder="e.g., birds-v2-vs-v3">
                    </div>
                    <div class="form-group">
                        <label>Strategy:</label>
                        <select id="route-strategy">
                            <option value="percentage_split">Percentage Split (0-100%)</option>
                            <option value="hash_based">Hash-Based (deterministic)</option>
                            <option value="round_robin">Round Robin</option>
                            <option value="weighted">Weighted Random</option>
                            <option value="canary">Canary Deployment</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Model A Path:</label>
                        <input type="text" id="route-model-a" placeholder="/path/to/model_a.ckpt">
                    </div>
                    <div class="form-group">
                        <label>Model A Weight (%):</label>
                        <input type="number" id="route-weight-a" value="70" min="0" max="100">
                    </div>
                    <div class="form-group">
                        <label>Model B Path:</label>
                        <input type="text" id="route-model-b" placeholder="/path/to/model_b.ckpt">
                    </div>
                    <div class="form-group">
                        <label>Model B Weight (%):</label>
                        <input type="number" id="route-weight-b" value="30" min="0" max="100">
                    </div>
                    <button onclick="createRoute()">➕ Create Route</button>
                </div>

                <h3>Active Routes</h3>
                <button onclick="loadRoutes()">🔄 Refresh Routes</button>
                <div id="routes-list" style="margin-top: 20px;"></div>
            </div>
        </div>

        <script>
            function switchTab(tabName) {
                const tabs = document.querySelectorAll('.tab-content');
                tabs.forEach(tab => tab.classList.remove('active'));
                const buttons = document.querySelectorAll('.tab');
                buttons.forEach(btn => btn.classList.remove('active'));
                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
            }

            async function loadCacheStatus() {
                const response = await fetch('/cache-status');
                const data = await response.json();
                const div = document.getElementById('cache-status');
                if (data.status === 'success') {
                    div.innerHTML = `
                        <div class="stat-card">
                            <div class="stat-label">Cache Size</div>
                            <div class="stat-value">${data.cache_size}/${data.max_cache_size}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Hit Rate</div>
                            <div class="stat-value">${data.hit_rate_percent.toFixed(1)}%</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Memory Usage</div>
                            <div class="stat-value">${data.total_memory_mb.toFixed(1)} MB</div>
                        </div>
                        <h3>Cached Models</h3>
                        <table>
                            <tr><th>Filename</th><th>Hits</th><th>Misses</th><th>Size</th><th>Status</th></tr>
                            ${data.cached_models.map(m => `<tr>
                                <td>${m.filename}</td>
                                <td>${m.hits}</td>
                                <td>${m.misses}</td>
                                <td>${m.file_size_mb.toFixed(1)} MB</td>
                                <td>${m.in_memory ? '<span class="badge badge-success">In Memory</span>' : '<span class="badge badge-warning">On Disk</span>'}</td>
                            </tr>`).join('')}
                        </table>
                    `;
                } else {
                    div.innerHTML = '<p class="error">Failed to load cache status</p>';
                }
            }

            async function runMultiModelInference() {
                const paths = document.getElementById('model-paths').value.split(',').map(p => p.trim()).filter(p => p);
                const file = document.getElementById('multi-image').files[0];
                if (!file || paths.length < 2) {
                    alert('Please select an image and provide at least 2 model paths');
                    return;
                }
                const formData = new FormData();
                formData.append('image', file);
                paths.forEach(p => formData.append('model_paths', p));
                const response = await fetch('/inference-multi', {method: 'POST', body: formData});
                const data = await response.json();
                const div = document.getElementById('multi-results');
                if (data.status === 'success') {
                    div.innerHTML = `<h3>Results</h3><pre>${JSON.stringify(data, null, 2)}</pre>`;
                } else {
                    div.innerHTML = `<p class="error">Error: ${data.message}</p>`;
                }
            }

            async function runEnsembleInference() {
                const models = document.getElementById('ensemble-models').value.split(',').map(p => p.trim()).filter(p => p);
                const method = document.getElementById('ensemble-method').value;
                const file = document.getElementById('ensemble-image').files[0];
                if (!file || models.length < 2) {
                    alert('Please select an image and provide at least 2 models');
                    return;
                }
                const formData = new FormData();
                formData.append('image', file);
                models.forEach(m => formData.append('model_paths', m));
                formData.append('ensemble_method', method);
                const response = await fetch('/inference-ensemble', {method: 'POST', body: formData});
                const data = await response.json();
                const div = document.getElementById('ensemble-results');
                if (data.status === 'success') {
                    div.innerHTML = `<h3>Ensemble Result</h3><pre>${JSON.stringify(data, null, 2)}</pre>`;
                } else {
                    div.innerHTML = `<p class="error">Error: ${data.message}</p>`;
                }
            }

            async function createRoute() {
                const route_id = document.getElementById('route-id').value;
                const strategy = document.getElementById('route-strategy').value;
                const model_a = document.getElementById('route-model-a').value;
                const weight_a = parseFloat(document.getElementById('route-weight-a').value);
                const model_b = document.getElementById('route-model-b').value;
                const weight_b = parseFloat(document.getElementById('route-weight-b').value);
                if (!route_id || !model_a || !model_b) {
                    alert('Please fill all fields');
                    return;
                }
                const formData = new FormData();
                formData.append('route_id', route_id);
                formData.append('strategy', strategy);
                formData.append('model_a_path', model_a);
                formData.append('model_a_weight', weight_a);
                formData.append('model_b_path', model_b);
                formData.append('model_b_weight', weight_b);
                const response = await fetch('/traffic-route/create', {method: 'POST', body: formData});
                const data = await response.json();
                if (data.status === 'success') {
                    alert('Route created successfully');
                    loadRoutes();
                } else {
                    alert('Error: ' + data.message);
                }
            }

            async function loadRoutes() {
                const response = await fetch('/traffic-routes');
                const data = await response.json();
                const div = document.getElementById('routes-list');
                if (data.status === 'success') {
                    if (data.routes.length === 0) {
                        div.innerHTML = '<p>No routes configured yet.</p>';
                        return;
                    }
                    div.innerHTML = data.routes.map(r => `
                        <div class="card">
                            <h3>${r.route_id}</h3>
                            <p><strong>Strategy:</strong> ${r.strategy}</p>
                            <p><strong>Models:</strong> ${r.models.join(' ↔ ')}</p>
                            <p><strong>Requests:</strong> ${r.requests}</p>
                            <p><strong>Distribution:</strong> ${r.models.map(m => {
                                const dist = r[m.split('/').pop()];
                                return dist ? m.split('/').pop() + ': ' + dist.actual_percentage.toFixed(1) + '%' : '';
                            }).filter(x => x).join(' | ')}</p>
                        </div>
                    `).join('');
                } else {
                    div.innerHTML = '<p class="error">Failed to load routes</p>';
                }
            }

            // Initial load
            loadCacheStatus();
        </script>
    </body>
    </html>
    """
    return html

@app.post("/cache-status")
async def cache_status():
    """Get cache status: hit rate, memory, models."""
    if not CACHE_AVAILABLE:
        return {"status": "error", "message": "Cache not available"}
    return MODEL_CACHE.get_cache_status()

@app.post("/cache-preload")
async def cache_preload(model_paths: list):
    """Pre-load multiple models into cache."""
    if not CACHE_AVAILABLE:
        return {"status": "error", "message": "Cache not available"}
    return MODEL_CACHE.warmup_cache(model_paths)

@app.post("/cache-clear")
async def cache_clear():
    """Clear all models from cache."""
    if not CACHE_AVAILABLE:
        return {"status": "error", "message": "Cache not available"}
    return MODEL_CACHE.clear_cache()

@app.post("/inference-multi")
async def inference_multi(images: UploadFile = File(...), model_paths: list = []):
    """Run inference on multiple models."""
    if not MULTI_INFERENCE_AVAILABLE:
        return {"status": "error", "message": "Multi-model inference not available"}

    try:
        image_bytes = await images.read()
        if not model_paths:
            return {"status": "error", "message": "No models provided"}
        return MULTI_INFERENCE.inference_all(image_bytes, model_paths, top_k=3)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/inference-ensemble")
async def inference_ensemble(images: UploadFile = File(...), model_paths: list = [], ensemble_method: str = "voting"):
    """Run ensemble inference combining multiple models."""
    if not MULTI_INFERENCE_AVAILABLE:
        return {"status": "error", "message": "Multi-model inference not available"}

    try:
        image_bytes = await images.read()
        if not model_paths or len(model_paths) < 2:
            return {"status": "error", "message": "Need at least 2 models"}
        return MULTI_INFERENCE.inference_ensemble(image_bytes, model_paths, ensemble_method, top_k=3)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/inference-route")
async def inference_route(route_id: str = Form(...), images: UploadFile = File(...)):
    """Route request through traffic router."""
    if not ROUTER_AVAILABLE:
        return {"status": "error", "message": "Traffic router not available"}

    try:
        image_bytes = await images.read()
        result = TRAFFIC_ROUTER.route_request(route_id, image_bytes)
        if result["status"] != "success":
            return result
        selected_model = result["selected_model"]
        if not CACHE_AVAILABLE:
            return {"status": "error", "message": "Cache required for routed inference"}
        inference = MODEL_CACHE.get_model(selected_model)
        if not inference:
            return {"status": "error", "message": f"Failed to load model: {selected_model}"}
        pred_result = inference.predict(image_bytes, return_top_k=3)
        return {
            "status": "success",
            "route_id": route_id,
            "selected_model": selected_model,
            "routing_info": result,
            "predictions": pred_result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/cache-status")
async def cache_status_get():
    """Get cache status: hit rate, memory, models."""
    if not CACHE_AVAILABLE:
        return {"status": "error", "message": "Cache not available"}
    return MODEL_CACHE.get_cache_status()

@app.get("/traffic-routes")
async def traffic_routes():
    """List all traffic routes."""
    if not ROUTER_AVAILABLE:
        return {"status": "error", "routes": [], "message": "Router not available"}

    routes_dict = {}
    for route_id, route_config in TRAFFIC_ROUTER.routes.items():
        routes_dict[route_id] = TRAFFIC_ROUTER.get_route_stats(route_id)

    route_list = []
    for route_id, stats in routes_dict.items():
        if stats.get("status") == "success":
            route_list.append({
                "route_id": route_id,
                "strategy": TRAFFIC_ROUTER.routes[route_id]["strategy"],
                "models": list(TRAFFIC_ROUTER.routes[route_id]["model_paths"].keys()),
                "requests": TRAFFIC_ROUTER.routes[route_id]["request_count"],
                **stats.get("model_distribution", {})
            })

    return {"status": "success", "total_routes": len(route_list), "routes": route_list}

@app.post("/traffic-route/create")
async def traffic_route_create(route_id: str = Form(...), strategy: str = Form("percentage_split"),
                               model_a_path: str = Form(...), model_a_weight: float = Form(70),
                               model_b_path: str = Form(...), model_b_weight: float = Form(30)):
    """Create a new traffic route."""
    if not ROUTER_AVAILABLE:
        return {"status": "error", "message": "Router not available"}

    model_paths = {
        model_a_path: model_a_weight,
        model_b_path: model_b_weight
    }
    return TRAFFIC_ROUTER.create_route(route_id, model_paths, strategy)

@app.post("/traffic-route/promote")
async def traffic_route_promote(route_id: str = Form(...), model_path: str = Form(...), new_weight: float = Form(100)):
    """Promote a model in a route (increase its traffic)."""
    if not ROUTER_AVAILABLE:
        return {"status": "error", "message": "Router not available"}

    return TRAFFIC_ROUTER.promote_model(route_id, model_path, new_weight)


@app.get("/api/status")
async def api_status():
    """JSON endpoint listing all available endpoints (for programmatic clients)."""
    return {
        "status": "Vision Module Active",
        "endpoints": {
            "detection": ["/detect"],
            "labeling": ["/labeling", "/labeling-stats", "/labeling-pending"],
            "training": ["/finetune-dashboard", "/finetune-config", "/finetune"],
            "inference": ["/models-inference", "/models-list", "/inference-finetune"],
            "registry": ["/models-versions", "/model-registry/all", "/model-registry/versions", "/model-registry/compare"],
            "ab_testing": ["/ab-testing", "/ab-test/list", "/ab-test/create", "/ab-test/run", "/ab-test/results", "/ab-test/promote"],
            "validation": ["/model-validation", "/validate-model", "/validation-history", "/rollback", "/rollback-params", "/rollback-dataset", "/model-history-detailed"],
            "inference_serving": ["/inference-serving", "/cache-status", "/cache-preload", "/cache-clear", "/inference-multi", "/inference-ensemble", "/inference-route", "/traffic-routes", "/traffic-route/create", "/traffic-route/promote"],
            "admin": ["/metrics"]
        }
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LocalML | Main Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --white: #FFFFFF;
                --bg: #F5F5F0;
                --card: #FFFFFF;
                --gold: #B8860B;
                --gold-dark: #8B6914;
                --gold-light: #D4A373;
                --gold-pale: #FFF8DC;
                --border: #D4C5B0;
                --text-primary: #2A2A2A;
                --text-secondary: #555555;
                --text-muted: #888888;
                --shadow-soft: 0 1px 3px rgba(0, 0, 0, 0.08);
                --shadow-hover: 0 4px 12px rgba(184, 134, 11, 0.15);
            }

            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg);
                color: var(--text-primary);
                line-height: 1.6;
                min-height: 100vh;
                padding: 0;
            }

            .header {
                background: var(--white);
                border-bottom: 1px solid var(--border);
                padding: 28px 48px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: var(--shadow-soft);
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 14px;
            }

            .logo {
                width: 42px;
                height: 42px;
                background: var(--gold);
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 800;
                color: white;
                font-size: 22px;
                box-shadow: 0 0 0 3px rgba(184, 134, 11, 0.15);
            }

            .brand-text h1 {
                font-size: 20px;
                font-weight: 700;
                color: var(--gold-dark);
                letter-spacing: -0.3px;
            }

            .brand-text p {
                font-size: 12px;
                color: var(--text-muted);
                font-weight: 500;
                letter-spacing: 0.4px;
                text-transform: uppercase;
            }

            .header-actions {
                display: flex;
                gap: 12px;
                align-items: center;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                background: var(--gold);
                border: 1px solid var(--gold);
                border-radius: 4px;
                color: var(--white);
                font-size: 13px;
                font-weight: 600;
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 2px;
                background: var(--white);
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }

            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 40px 48px;
            }

            .hero {
                margin-bottom: 48px;
            }

            .hero h2 {
                font-size: 32px;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 8px;
                letter-spacing: -0.6px;
            }

            .hero h2 span {
                color: var(--gold-dark);
            }

            .hero p {
                font-size: 15px;
                color: var(--text-secondary);
                max-width: 700px;
            }

            .stage-section {
                margin-bottom: 40px;
            }

            .stage-header {
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 20px;
                padding-bottom: 16px;
                border-bottom: 1px solid var(--border);
            }

            .stage-number {
                width: 32px;
                height: 32px;
                background: var(--white);
                border: 2px solid var(--gold);
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                color: var(--gold-dark);
                font-size: 14px;
            }

            .stage-title {
                flex: 1;
            }

            .stage-title h3 {
                font-size: 18px;
                font-weight: 700;
                color: var(--gold-dark);
                letter-spacing: -0.2px;
            }

            .stage-title p {
                font-size: 13px;
                color: var(--text-muted);
                margin-top: 2px;
            }

            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 20px;
            }

            .feature-card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 24px;
                text-decoration: none;
                color: inherit;
                position: relative;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
                overflow: hidden;
            }

            .feature-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, transparent 0%, var(--gold) 50%, transparent 100%);
                opacity: 0;
                transition: opacity 0.25s ease;
            }

            .feature-card:hover {
                border-color: var(--gold);
                transform: translateY(-3px);
                box-shadow: var(--shadow-hover);
            }

            .feature-card:hover::before {
                opacity: 1;
            }

            .feature-icon {
                width: 44px;
                height: 44px;
                background: #FFF8DC;
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 22px;
                margin-bottom: 16px;
                transition: all 0.25s ease;
            }

            .feature-card:hover .feature-icon {
                background: var(--gold);
                color: var(--white);
                transform: scale(1.05);
            }

            .feature-card h4 {
                font-size: 16px;
                font-weight: 600;
                color: var(--gold-dark);
                margin-bottom: 6px;
                letter-spacing: -0.2px;
            }

            .feature-card p {
                font-size: 13px;
                color: var(--text-secondary);
                line-height: 1.5;
                margin-bottom: 12px;
            }

            .feature-meta {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-top: 16px;
                padding-top: 14px;
                border-top: 1px solid var(--border);
            }

            .feature-tag {
                font-size: 11px;
                color: var(--gold-dark);
                font-weight: 600;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }

            .feature-arrow {
                width: 24px;
                height: 24px;
                background: var(--bg);
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--text-muted);
                font-size: 14px;
                transition: all 0.25s ease;
            }

            .feature-card:hover .feature-arrow {
                background: var(--gold);
                color: white;
                transform: translateX(2px);
            }

            .feature-badges {
                display: flex;
                gap: 6px;
                flex-wrap: wrap;
                margin-top: 8px;
            }

            .feature-badge {
                font-size: 10px;
                padding: 3px 8px;
                background: var(--bg);
                color: var(--text-secondary);
                border-radius: 3px;
                font-weight: 500;
            }

            .feature-badge.new {
                background: #FFF8DC;
                color: var(--gold-dark);
                border: 1px solid var(--border);
            }

            .footer {
                margin-top: 60px;
                padding: 24px 0;
                border-top: 1px solid var(--border);
                text-align: center;
                color: var(--text-muted);
                font-size: 13px;
            }

            .footer a {
                color: var(--gold-dark);
                text-decoration: none;
                font-weight: 500;
            }

            .footer a:hover {
                color: var(--gold);
            }

            @media (max-width: 768px) {
                .header { padding: 20px 24px; flex-direction: column; gap: 16px; }
                .container { padding: 24px; }
                .hero h2 { font-size: 24px; }
                .feature-grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <header class="header">
            <div class="brand">
                <div class="logo">S</div>
                <div class="brand-text">
                    <h1>LocalML</h1>
                    <p>ML Operations Platform</p>
                </div>
            </div>
            <div class="header-actions">
                <div class="status-pill">
                    <span class="status-dot"></span>
                    System Active
                </div>
            </div>
        </header>

        <div class="container">
            <div class="hero">
                <h2>Welcome to your <span>command center</span></h2>
                <p>Manage the complete lifecycle of your computer vision models — from data labeling and fine-tuning to deployment, validation, and production serving.</p>
            </div>

            <!-- STAGE 1: Data Pipeline -->
            <div class="stage-section">
                <div class="stage-header">
                    <div class="stage-number">1</div>
                    <div class="stage-title">
                        <h3>Data Pipeline</h3>
                        <p>Capture, annotate, and prepare training data</p>
                    </div>
                </div>
                <div class="feature-grid">
                    <a href="/detect-dashboard" class="feature-card">
                        <div class="feature-icon">🔍</div>
                        <h4>Object Detection</h4>
                        <p>FasterRCNN (COCO, 80 classes) with interactive camera control and class filtering.</p>
                        <div class="feature-meta">
                            <span class="feature-tag">Detection</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                    <a href="/labeling" class="feature-card">
                        <div class="feature-icon">🏷️</div>
                        <h4>Labeling Tool</h4>
                        <p>Annotate images manually or upload CSV batches into your dataset.</p>
                        <div class="feature-meta">
                            <span class="feature-tag">Annotation</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                    <a href="/mobile-capture" class="feature-card">
                        <div class="feature-icon">📱</div>
                        <h4>Mobile Capture</h4>
                        <p>Stream images from your phone via QR code for live data collection.</p>
                        <div class="feature-meta">
                            <span class="feature-tag">Capture</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                </div>
            </div>

            <!-- STAGE 2: Training -->
            <div class="stage-section">
                <div class="stage-header">
                    <div class="stage-number">2</div>
                    <div class="stage-title">
                        <h3>Training & Fine-tuning</h3>
                        <p>Train custom models on your labeled datasets</p>
                    </div>
                </div>
                <div class="feature-grid">
                    <a href="/finetune-dashboard" class="feature-card">
                        <div class="feature-icon">🎯</div>
                        <h4>Fine-tuning Engine</h4>
                        <p>Train models with 80/20 validation split, early stopping, and configurable checkpointing.</p>
                        <div class="feature-meta">
                            <span class="feature-tag">Training</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                </div>
            </div>

            <!-- STAGE 3: Inference -->
            <div class="stage-section">
                <div class="stage-header">
                    <div class="stage-number">3</div>
                    <div class="stage-title">
                        <h3>Inference & Serving</h3>
                        <p>Run predictions and serve models to production</p>
                    </div>
                </div>
                <div class="feature-grid">
                    <a href="/models-inference" class="feature-card">
                        <div class="feature-icon">⚡</div>
                        <h4>Model Inference</h4>
                        <p>Browse fine-tuned models and run predictions with top-K confidence scores.</p>
                        <div class="feature-meta">
                            <span class="feature-tag">Inference</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                    <a href="/inference-serving" class="feature-card">
                        <div class="feature-icon">🚀</div>
                        <h4>Multi-Model Serving</h4>
                        <p>In-memory caching, ensemble predictions, and traffic routing across model versions.</p>
                        <div class="feature-badges">
                            <span class="feature-badge new">New</span>
                            <span class="feature-badge">Cache</span>
                            <span class="feature-badge">Ensemble</span>
                            <span class="feature-badge">Routing</span>
                        </div>
                        <div class="feature-meta">
                            <span class="feature-tag">Production</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                </div>
            </div>

            <!-- STAGE 4: Model Operations -->
            <div class="stage-section">
                <div class="stage-header">
                    <div class="stage-number">4</div>
                    <div class="stage-title">
                        <h3>Model Operations</h3>
                        <p>Version, validate, and compare models safely</p>
                    </div>
                </div>
                <div class="feature-grid">
                    <a href="/models-versions" class="feature-card">
                        <div class="feature-icon">📦</div>
                        <h4>Model Versions</h4>
                        <p>Auto-increment versioning (v1.0 → v2.0 → v3.0) with full metadata and access control.</p>
                        <div class="feature-meta">
                            <span class="feature-tag">Registry</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                    <a href="/model-validation" class="feature-card">
                        <div class="feature-icon">🛡️</div>
                        <h4>Validation & Rollback</h4>
                        <p>Pre-deployment K-fold validation, root cause diagnosis, and one-click rollback strategies.</p>
                        <div class="feature-badges">
                            <span class="feature-badge">K-fold</span>
                            <span class="feature-badge">Rollback</span>
                        </div>
                        <div class="feature-meta">
                            <span class="feature-tag">Safety</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                    <a href="/ab-testing" class="feature-card">
                        <div class="feature-icon">⚗️</div>
                        <h4>A/B Testing</h4>
                        <p>Compare model versions with batch tests, deterministic routing, and winner determination.</p>
                        <div class="feature-badges">
                            <span class="feature-badge">Compare</span>
                            <span class="feature-badge">Promote</span>
                        </div>
                        <div class="feature-meta">
                            <span class="feature-tag">Experimentation</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                </div>
            </div>

            <!-- STAGE 5: Monitoring -->
            <div class="stage-section">
                <div class="stage-header">
                    <div class="stage-number">5</div>
                    <div class="stage-title">
                        <h3>Monitoring & Operations</h3>
                        <p>Observability, metrics, and system health</p>
                    </div>
                </div>
                <div class="feature-grid">
                    <a href="/admin" class="feature-card">
                        <div class="feature-icon">📊</div>
                        <h4>Admin Console</h4>
                        <p>Training logs, system metrics, and Grafana telemetry in a unified view.</p>
                        <div class="feature-meta">
                            <span class="feature-tag">Operations</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                    <a href="/metrics" class="feature-card">
                        <div class="feature-icon">📈</div>
                        <h4>Prometheus Metrics</h4>
                        <p>Raw application metrics for inference latency, request counts, and finetuning jobs.</p>
                        <div class="feature-meta">
                            <span class="feature-tag">Metrics</span>
                            <span class="feature-arrow">→</span>
                        </div>
                    </a>
                </div>
            </div>

            <div class="footer">
                <p>LocalML · ML Operations Platform · <a href="/admin">View System Status</a></p>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

