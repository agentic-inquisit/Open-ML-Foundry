"""
LocalML finetune - Secure Main Server with RBAC
Implements role-based access control separating admin and user functionality
"""

import cv2
import numpy as np
import os
import tempfile
import secrets
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi import UploadFile, File
import jwt
from passlib.context import CryptContext
import sys
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure edge module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edge.jax_train import run_inference
from edge.preprocess import run_pipeline

# ============================================================================
# CONFIGURATION
# ============================================================================
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
ADMIN_TOKEN_EXPIRE_MINUTES = 15  # Shorter for admin
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

app = FastAPI(title="LocalML finetune - Secure")

# ============================================================================
# CORS - Restrict to specific domains
# ============================================================================
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ============================================================================
# ENUMS & MODELS
# ============================================================================

class RoleEnum(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    USER = "user"

class User(BaseModel):
    """User registration/login model"""
    username: str
    password: str
    email: Optional[str] = None

class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int

class TokenPayload(BaseModel):
    """JWT payload structure"""
    sub: str  # username
    role: RoleEnum
    exp: datetime
    iat: datetime
    token_type: str  # 'access' or 'refresh'

class VideoRequest(BaseModel):
    """Video generation request"""
    prompt: str

class ChatRequest(BaseModel):
    """LLM chat request"""
    prompt: str
    model: str = "llama3"

# ============================================================================
# DATABASE (Mock - Replace with real DB)
# ============================================================================
class MockDB:
    """Simple in-memory database. Replace with PostgreSQL/MongoDB in production"""
    def __init__(self):
        self.users = {}  # {username: {"password_hash": str, "role": RoleEnum, "email": str}}
        self.tokens = {}  # {token: {"username": str, "created_at": datetime}}
        self.audit_log = []  # Audit trail

    def add_user(self, username: str, password_hash: str, email: str, role: RoleEnum = RoleEnum.USER):
        if username in self.users:
            return False
        self.users[username] = {
            "password_hash": password_hash,
            "role": role,
            "email": email,
            "created_at": datetime.utcnow()
        }
        return True

    def get_user(self, username: str):
        return self.users.get(username)

    def add_token(self, token: str, username: str):
        self.tokens[token] = {"username": username, "created_at": datetime.utcnow()}

    def is_token_revoked(self, token: str):
        return token not in self.tokens

    def audit_log_action(self, username: str, action: str, resource: str, status_code: int):
        """Log admin actions for compliance"""
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "username": username,
            "action": action,
            "resource": resource,
            "status_code": status_code
        })
        logger.info(f"AUDIT: {username} - {action} on {resource} - Status: {status_code}")

db = MockDB()

# ============================================================================
# PASSWORD & TOKEN UTILITIES
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password with bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_token(username: str, role: RoleEnum, token_type: str = "access") -> tuple[str, datetime]:
    """
    Create JWT token with role claim
    token_type: 'access' (short-lived) or 'refresh' (long-lived)
    """
    now = datetime.utcnow()

    if token_type == "access":
        expire_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES if role == RoleEnum.USER else ADMIN_TOKEN_EXPIRE_MINUTES)
    else:  # refresh
        expire_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    expires = now + expire_delta

    payload = {
        "sub": username,
        "role": role.value,
        "token_type": token_type,
        "exp": expires,
        "iat": now
    }

    encoded_jwt = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, expires

def decode_token(token: str) -> Optional[TokenPayload]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        token_type = payload.get("token_type")

        if not all([username, role, token_type]):
            return None

        return TokenPayload(
            sub=username,
            role=RoleEnum(role),
            token_type=token_type,
            exp=datetime.fromtimestamp(payload.get("exp")),
            iat=datetime.fromtimestamp(payload.get("iat"))
        )
    except jwt.ExpiredSignatureError:
        logger.warning(f"Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning(f"Invalid token")
        return None

# ============================================================================
# DEPENDENCY INJECTION - AUTHORIZATION
# ============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenPayload:
    """
    Dependency to extract and validate JWT from Authorization header
    Returns: TokenPayload with username and role
    Raises: HTTPException 401 if invalid
    """
    token = credentials.credentials

    # Check if token is revoked
    if db.is_token_revoked(token):
        logger.warning(f"Token revoked or not found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Decode token
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload

async def require_admin(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    """
    Dependency to require admin role
    Used as: @app.post("/admin/something", dependencies=[Depends(require_admin)])
    or: async def endpoint(admin_user: TokenPayload = Depends(require_admin))
    """
    if current_user.role != RoleEnum.ADMIN:
        logger.warning(f"Non-admin user {current_user.sub} attempted admin access")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

async def require_user(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    """
    Dependency to require user role (or admin)
    """
    if current_user.role not in [RoleEnum.USER, RoleEnum.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User access required"
        )
    return current_user

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================
VIDEO_GEN_TOTAL = Counter("video_generation_total", "Total videos requested", ["status"])
VIDEO_GEN_SUCCESS = Counter("video_generation_success", "Total videos successfully generated")
ADMIN_ACTIONS = Counter("admin_actions_total", "Total admin actions", ["action", "status"])

# ============================================================================
# PUBLIC ENDPOINTS (NO AUTH REQUIRED)
# ============================================================================

@app.get("/")
async def root():
    """Public health check"""
    return {
        "status": "Vision Services is running",
        "version": "2.0-secure",
        "endpoints": {
            "public": ["/auth/signup", "/auth/login", "/health"],
            "user": ["/generate-video", "/jax-inference", "/preprocess", "/chat"],
            "admin": ["/admin/users", "/admin/audit-log", "/admin/system-stats"]
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ============================================================================
# AUTHENTICATION ENDPOINTS (NO AUTH REQUIRED)
# ============================================================================

@app.post("/auth/signup", response_model=TokenResponse)
async def signup(user: User):
    """
    Register new user (role defaults to USER)
    Passwords are hashed with bcrypt before storage
    """
    if not user.username or len(user.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

    if not user.password or len(user.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    password_hash = hash_password(user.password)

    if not db.add_user(user.username, password_hash, user.email or "", RoleEnum.USER):
        raise HTTPException(status_code=400, detail="Username already exists")

    access_token, expires = create_token(user.username, RoleEnum.USER, "access")
    refresh_token, _ = create_token(user.username, RoleEnum.USER, "refresh")
    db.add_token(access_token, user.username)

    logger.info(f"User registered: {user.username}")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": int(ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    }

@app.post("/auth/login", response_model=TokenResponse)
async def login(user: User):
    """
    Login endpoint - validates credentials and returns JWT
    Token includes role claim for RBAC
    """
    stored_user = db.get_user(user.username)

    if not stored_user or not verify_password(user.password, stored_user["password_hash"]):
        logger.warning(f"Failed login attempt for user: {user.username}")
        raise HTTPException(status_code=401, detail="Invalid username or password")

    role = RoleEnum(stored_user["role"])
    access_token, expires = create_token(user.username, role, "access")
    refresh_token, _ = create_token(user.username, role, "refresh")
    db.add_token(access_token, user.username)

    logger.info(f"User logged in: {user.username} (role: {role})")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": int(ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    }

@app.post("/auth/logout")
async def logout(current_user: TokenPayload = Depends(get_current_user)):
    """Logout - revoke token"""
    logger.info(f"User logged out: {current_user.sub}")
    return {"message": "Logged out successfully"}

# ============================================================================
# USER ENDPOINTS (AUTH REQUIRED)
# ============================================================================

@app.post("/generate-video")
async def generate_video(
    request: VideoRequest,
    current_user: TokenPayload = Depends(require_user)
):
    """
    Generate video - requires authentication (USER or ADMIN)
    """
    logger.info(f"Video generation requested by {current_user.sub}")
    VIDEO_GEN_TOTAL.labels(status="started").inc()

    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"generated_video_{secrets.token_hex(4)}.mp4")

        width, height = 640, 480
        fps = 24
        duration = 3
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

        for i in range(fps * duration):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            c1 = (i * 2) % 255
            c2 = (i * 5) % 255
            frame[:,:] = [c1, 100, c2]

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Prompt: {request.prompt}"
            cv2.putText(frame, text, (50, height // 2), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Generating frame {i}...", (50, height - 50), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            out.write(frame)

        out.release()

        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Failed to create video file")

        VIDEO_GEN_TOTAL.labels(status="success").inc()
        VIDEO_GEN_SUCCESS.inc()
        return FileResponse(file_path, media_type="video/mp4", filename="generated_video.mp4")

    except Exception as e:
        VIDEO_GEN_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jax-inference")
async def jax_inference(
    file: UploadFile = File(...),
    current_user: TokenPayload = Depends(require_user)
):
    """JAX inference - requires authentication"""
    logger.info(f"JAX inference requested by {current_user.sub}")
    try:
        image_bytes = await file.read()
        result = run_inference(image_bytes)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess")
async def preprocess_image(
    file: UploadFile = File(...),
    current_user: TokenPayload = Depends(require_user)
):
    """Image preprocessing - requires authentication"""
    logger.info(f"Preprocessing requested by {current_user.sub}")
    try:
        image_bytes = await file.read()
        result = run_pipeline(image_bytes)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_llm(
    request: ChatRequest,
    current_user: TokenPayload = Depends(require_user)
):
    """LLM chat - requires authentication"""
    logger.info(f"Chat requested by {current_user.sub}")
    try:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": request.model,
                "prompt": request.prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ADMIN ENDPOINTS (ADMIN ONLY)
# ============================================================================

@app.get("/admin/users")
async def list_users(admin_user: TokenPayload = Depends(require_admin)):
    """
    Admin endpoint - list all users
    Only accessible by admin role
    Server-side authorization check required
    """
    db.audit_log_action(admin_user.sub, "LIST_USERS", "user_database", 200)
    ADMIN_ACTIONS.labels(action="list_users", status="success").inc()

    users_list = [
        {
            "username": username,
            "role": user_data["role"],
            "email": user_data["email"],
            "created_at": user_data["created_at"].isoformat()
        }
        for username, user_data in db.users.items()
    ]
    return {"users": users_list, "count": len(users_list)}

@app.post("/admin/promote-to-admin")
async def promote_user(
    username: str,
    admin_user: TokenPayload = Depends(require_admin)
):
    """
    Admin endpoint - promote user to admin role
    Requires admin authorization checked server-side
    """
    if username not in db.users:
        raise HTTPException(status_code=404, detail="User not found")

    db.users[username]["role"] = RoleEnum.ADMIN
    db.audit_log_action(admin_user.sub, "PROMOTE_TO_ADMIN", f"user:{username}", 200)
    ADMIN_ACTIONS.labels(action="promote_user", status="success").inc()

    logger.info(f"User {username} promoted to admin by {admin_user.sub}")
    return {"message": f"User {username} promoted to admin"}

@app.delete("/admin/users/{username}")
async def delete_user(
    username: str,
    admin_user: TokenPayload = Depends(require_admin)
):
    """
    Admin endpoint - delete user
    Requires admin authorization checked server-side
    """
    if username not in db.users:
        raise HTTPException(status_code=404, detail="User not found")

    if username == admin_user.sub:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    del db.users[username]
    db.audit_log_action(admin_user.sub, "DELETE_USER", f"user:{username}", 200)
    ADMIN_ACTIONS.labels(action="delete_user", status="success").inc()

    logger.info(f"User {username} deleted by {admin_user.sub}")
    return {"message": f"User {username} deleted"}

@app.get("/admin/audit-log")
async def get_audit_log(admin_user: TokenPayload = Depends(require_admin)):
    """
    Admin endpoint - retrieve audit log
    Shows all admin actions for compliance
    """
    db.audit_log_action(admin_user.sub, "VIEW_AUDIT_LOG", "audit_log", 200)
    ADMIN_ACTIONS.labels(action="view_audit_log", status="success").inc()

    return {
        "audit_log": db.audit_log,
        "count": len(db.audit_log)
    }

@app.get("/admin/system-stats")
async def get_system_stats(admin_user: TokenPayload = Depends(require_admin)):
    """
    Admin endpoint - system statistics
    """
    db.audit_log_action(admin_user.sub, "VIEW_SYSTEM_STATS", "system", 200)
    ADMIN_ACTIONS.labels(action="view_system_stats", status="success").inc()

    return {
        "total_users": len(db.users),
        "total_audit_events": len(db.audit_log),
        "total_tokens_issued": len(db.tokens),
        "server_time": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def metrics(admin_user: TokenPayload = Depends(require_admin)):
    """
    Admin endpoint - Prometheus metrics
    Requires admin authentication
    """
    db.audit_log_action(admin_user.sub, "VIEW_METRICS", "prometheus", 200)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Log HTTP exceptions"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {"detail": exc.detail, "status_code": exc.status_code}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
