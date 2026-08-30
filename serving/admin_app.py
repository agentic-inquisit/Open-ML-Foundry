"""
LocalML finetune - Admin Application
Separate domain: admin.example.com or localhost:8001
Only admin features, user features NOT accessible
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import jwt
from passlib.context import CryptContext
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ADMIN_TOKEN_EXPIRE_MINUTES = 15  # Shorter TTL for admin

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

app = FastAPI(
    title="LocalML finetune - Admin App",
    description="Admin-only application (separate domain from user app)",
    docs_url="/docs",  # Only accessible on admin domain
)

# ============================================================================
# CORS - Admin domain only (strict)
# ============================================================================
ALLOWED_ADMIN_ORIGINS = [
    "http://localhost:3001",  # Different port/domain from user app
    os.getenv("ADMIN_FRONTEND_URL", "http://localhost:3001"),
]

# Optional: IP whitelist for admin domain
ADMIN_IP_WHITELIST = os.getenv("ADMIN_IP_WHITELIST", "").split(",") if os.getenv("ADMIN_IP_WHITELIST") else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ADMIN_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# ============================================================================
# MODELS
# ============================================================================

class RoleEnum(str, Enum):
    ADMIN = "admin"
    USER = "user"

class AdminCredentials(BaseModel):
    """Admin login credentials"""
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenPayload(BaseModel):
    sub: str
    role: RoleEnum
    exp: datetime
    iat: datetime

# ============================================================================
# DATABASE - Shared with user domain
# ============================================================================
class SharedAuthDB:
    """Shared authentication database (accessible from both domains)"""

    def __init__(self):
        self.users = {}
        self.audit_log = []

    def get_user(self, username: str):
        return self.users.get(username)

    def audit_log_action(self, username: str, action: str, resource: str, status_code: int):
        """Log admin actions"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "username": username,
            "action": action,
            "resource": resource,
            "status_code": status_code
        }
        self.audit_log.append(entry)
        logger.info(f"AUDIT: {username} - {action} on {resource} - Status: {status_code}")

    def promote_user_to_admin(self, username: str) -> bool:
        if username not in self.users:
            return False
        self.users[username]["role"] = RoleEnum.ADMIN
        return True

    def delete_user(self, username: str) -> bool:
        if username not in self.users:
            return False
        del self.users[username]
        return True

shared_db = SharedAuthDB()

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================
ADMIN_ACTIONS = Counter("admin_actions_total", "Total admin actions", ["action", "status"])
AUTH_ATTEMPTS = Counter("admin_auth_attempts", "Admin auth attempts", ["result"])

# ============================================================================
# TOKEN UTILITIES
# ============================================================================

def decode_token(token: str) -> Optional[TokenPayload]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")

        if not all([username, role]):
            return None

        return TokenPayload(
            sub=username,
            role=RoleEnum(role),
            exp=datetime.fromtimestamp(payload.get("exp")),
            iat=datetime.fromtimestamp(payload.get("iat"))
        )
    except jwt.ExpiredSignatureError:
        logger.warning("Admin token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        return None

def create_admin_token(username: str) -> tuple[str, int]:
    """Create JWT token for admin (shorter expiration)"""
    now = datetime.utcnow()
    expires = now + timedelta(minutes=ADMIN_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": username,
        "role": RoleEnum.ADMIN.value,
        "exp": expires,
        "iat": now
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token, int(ADMIN_TOKEN_EXPIRE_MINUTES * 60)

# ============================================================================
# DEPENDENCY INJECTION - AUTHORIZATION
# ============================================================================

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenPayload:
    """
    Extract and validate JWT from Authorization header
    CRITICAL: Verify role is ADMIN before allowing any action
    """
    token = credentials.credentials

    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    # ◄── CRITICAL: Verify ADMIN role server-side
    if payload.role != RoleEnum.ADMIN:
        logger.warning(f"Non-admin user {payload.sub} attempted admin access")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )

    return payload

# ============================================================================
# PUBLIC ENDPOINTS (NO AUTH REQUIRED)
# ============================================================================

@app.get("/")
async def root():
    """Admin domain root"""
    return {
        "app": "Admin Application",
        "domain": "admin.example.com",
        "status": "running",
        "endpoints": {
            "public": ["/health", "/auth/admin-login"],
            "admin": ["/users", "/promote", "/delete-user", "/audit-log", "/system-stats", "/metrics"]
        },
        "note": "User features are NOT available on this domain. Use app.example.com"
    }

@app.get("/health")
async def health_check():
    """Health check for load balancers"""
    return {"status": "healthy", "app": "admin", "timestamp": datetime.utcnow().isoformat()}

# ============================================================================
# AUTHENTICATION ENDPOINTS (ADMIN ONLY)
# ============================================================================

@app.post("/auth/admin-login", response_model=TokenResponse)
async def admin_login(credentials: AdminCredentials):
    """
    Admin login endpoint
    Only allows users with ADMIN role to login
    """
    stored_user = shared_db.get_user(credentials.username)

    if not stored_user or stored_user["role"] != RoleEnum.ADMIN:
        logger.warning(f"Non-admin login attempt: {credentials.username}")
        AUTH_ATTEMPTS.labels(result="failed_not_admin").inc()
        raise HTTPException(status_code=401, detail="Invalid credentials or not admin")

    # Verify password
    if not pwd_context.verify(credentials.password, stored_user["password_hash"]):
        logger.warning(f"Failed admin login: {credentials.username}")
        AUTH_ATTEMPTS.labels(result="failed_password").inc()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create admin token
    access_token, expires_in = create_admin_token(credentials.username)

    logger.info(f"Admin logged in: {credentials.username}")
    AUTH_ATTEMPTS.labels(result="success").inc()

    return {
        "access_token": access_token,
        "expires_in": expires_in
    }

# ============================================================================
# ADMIN ENDPOINTS (ADMIN ONLY)
# ============================================================================

@app.get("/users")
async def list_users(admin_user: TokenPayload = Depends(get_current_admin)):
    """
    List all users
    ◄── CRITICAL: get_current_admin dependency validates admin role before this runs
    """
    shared_db.audit_log_action(admin_user.sub, "LIST_USERS", "user_database", 200)
    ADMIN_ACTIONS.labels(action="list_users", status="success").inc()

    users_list = [
        {
            "username": username,
            "role": user_data["role"],
            "email": user_data["email"],
            "created_at": user_data["created_at"].isoformat()
        }
        for username, user_data in shared_db.users.items()
    ]

    return {"users": users_list, "count": len(users_list)}

@app.post("/promote")
async def promote_user(
    username: str,
    admin_user: TokenPayload = Depends(get_current_admin)
):
    """
    Promote user to admin role
    ◄── CRITICAL: get_current_admin dependency validates admin role before this runs
    """
    if not shared_db.promote_user_to_admin(username):
        raise HTTPException(status_code=404, detail="User not found")

    shared_db.audit_log_action(admin_user.sub, "PROMOTE_TO_ADMIN", f"user:{username}", 200)
    ADMIN_ACTIONS.labels(action="promote_user", status="success").inc()

    logger.info(f"User {username} promoted to admin by {admin_user.sub}")
    return {"message": f"User {username} promoted to admin"}

@app.delete("/users/{username}")
async def delete_user(
    username: str,
    admin_user: TokenPayload = Depends(get_current_admin)
):
    """
    Delete user
    ◄── CRITICAL: get_current_admin dependency validates admin role before this runs
    """
    if username == admin_user.sub:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    if not shared_db.delete_user(username):
        raise HTTPException(status_code=404, detail="User not found")

    shared_db.audit_log_action(admin_user.sub, "DELETE_USER", f"user:{username}", 200)
    ADMIN_ACTIONS.labels(action="delete_user", status="success").inc()

    logger.info(f"User {username} deleted by {admin_user.sub}")
    return {"message": f"User {username} deleted"}

@app.get("/audit-log")
async def get_audit_log(admin_user: TokenPayload = Depends(get_current_admin)):
    """
    View audit log of all admin actions
    ◄── CRITICAL: get_current_admin dependency validates admin role before this runs
    """
    shared_db.audit_log_action(admin_user.sub, "VIEW_AUDIT_LOG", "audit_log", 200)
    ADMIN_ACTIONS.labels(action="view_audit_log", status="success").inc()

    return {
        "audit_log": shared_db.audit_log,
        "count": len(shared_db.audit_log)
    }

@app.get("/system-stats")
async def get_system_stats(admin_user: TokenPayload = Depends(get_current_admin)):
    """
    View system statistics
    ◄── CRITICAL: get_current_admin dependency validates admin role before this runs
    """
    shared_db.audit_log_action(admin_user.sub, "VIEW_SYSTEM_STATS", "system", 200)
    ADMIN_ACTIONS.labels(action="view_system_stats", status="success").inc()

    return {
        "total_users": len(shared_db.users),
        "total_audit_events": len(shared_db.audit_log),
        "server_time": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def metrics(admin_user: TokenPayload = Depends(get_current_admin)):
    """
    Prometheus metrics (admin only)
    ◄── CRITICAL: get_current_admin dependency validates admin role before this runs
    """
    shared_db.audit_log_action(admin_user.sub, "VIEW_METRICS", "prometheus", 200)
    return Response(generate_latest(), media_type="text/plain")

# ============================================================================
# NO USER ENDPOINTS ON ADMIN DOMAIN
# ============================================================================
# Notice: There are NO user endpoints here (generate-video, jax-inference, etc.)
# User endpoints are ONLY on user domain (app.example.com)
# This ensures complete architectural separation

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
