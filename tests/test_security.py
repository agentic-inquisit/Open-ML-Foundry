"""
Security Tests for RBAC Implementation

Run with: pytest test_security.py -v

These tests verify:
1. Authentication works correctly
2. Users cannot access admin endpoints
3. Admins can access admin endpoints
4. Audit logging works
5. Token validation works
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

# Import the secure app
from serving.main_secure import app, db, hash_password, RoleEnum

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_db():
    """Reset database before and after each test"""
    db.users.clear()
    db.tokens.clear()
    db.audit_log.clear()
    yield
    db.users.clear()
    db.tokens.clear()
    db.audit_log.clear()


# ============================================================================
# PUBLIC ENDPOINT TESTS
# ============================================================================

def test_health_check_public():
    """Public health endpoint should be accessible without auth"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint_public():
    """Root endpoint should be publicly accessible"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Vision Services is running" in response.json()["status"]


# ============================================================================
# SIGNUP/LOGIN TESTS
# ============================================================================

def test_signup_creates_user_successfully():
    """User should be able to sign up"""
    response = client.post(
        "/auth/signup",
        json={"username": "alice", "password": "SecurePass123!"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"
    assert "refresh_token" in response.json()


def test_signup_requires_minimum_password_length():
    """Password must be at least 8 characters"""
    response = client.post(
        "/auth/signup",
        json={"username": "alice", "password": "short"}
    )
    assert response.status_code == 400
    assert "at least 8 characters" in response.json()["detail"]


def test_signup_prevents_duplicate_username():
    """Cannot create two users with same username"""
    client.post(
        "/auth/signup",
        json={"username": "alice", "password": "SecurePass123!"}
    )
    response = client.post(
        "/auth/signup",
        json={"username": "alice", "password": "OtherPass123!"}
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_login_succeeds_with_correct_password():
    """User should be able to login with correct credentials"""
    # Signup
    client.post(
        "/auth/signup",
        json={"username": "alice", "password": "SecurePass123!"}
    )
    # Login
    response = client.post(
        "/auth/login",
        json={"username": "alice", "password": "SecurePass123!"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_fails_with_wrong_password():
    """Login should fail with incorrect password"""
    client.post(
        "/auth/signup",
        json={"username": "alice", "password": "SecurePass123!"}
    )
    response = client.post(
        "/auth/login",
        json={"username": "alice", "password": "WrongPass123!"}
    )
    assert response.status_code == 401
    assert "Invalid" in response.json()["detail"]


def test_login_fails_with_nonexistent_user():
    """Login should fail if user doesn't exist"""
    response = client.post(
        "/auth/login",
        json={"username": "nonexistent", "password": "SomePass123!"}
    )
    assert response.status_code == 401


# ============================================================================
# USER ENDPOINT TESTS
# ============================================================================

def test_user_endpoint_requires_auth():
    """User endpoints should require authentication"""
    response = client.post(
        "/generate-video",
        json={"prompt": "test"}
    )
    assert response.status_code == 403


def test_user_endpoint_rejects_invalid_token():
    """User endpoints should reject invalid tokens"""
    response = client.post(
        "/generate-video",
        json={"prompt": "test"},
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401


def test_user_endpoint_works_with_valid_token():
    """User endpoint should work with valid token"""
    # Signup to get token
    signup_response = client.post(
        "/auth/signup",
        json={"username": "alice", "password": "SecurePass123!"}
    )
    token = signup_response.json()["access_token"]

    # Access user endpoint
    response = client.post(
        "/generate-video",
        json={"prompt": "test video"},
        headers={"Authorization": f"Bearer {token}"}
    )
    # Note: May fail if cv2/video generation not available, but auth should pass
    assert response.status_code in [200, 500]  # Auth passes, may fail on processing


# ============================================================================
# ADMIN ENDPOINT TESTS - THE CRITICAL SECURITY TESTS
# ============================================================================

def test_user_cannot_access_admin_list_users_endpoint():
    """CRITICAL: Regular user should NOT be able to list users"""
    # Create regular user
    signup_response = client.post(
        "/auth/signup",
        json={"username": "alice", "password": "SecurePass123!"}
    )
    token = signup_response.json()["access_token"]

    # Try to access admin endpoint
    response = client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {token}"}
    )

    # MUST be 403 Forbidden (not 200!)
    assert response.status_code == 403
    assert "Admin access required" in response.json()["detail"]


def test_unauthenticated_cannot_access_admin_endpoint():
    """Admin endpoints should reject unauthenticated requests"""
    response = client.get("/admin/users")
    assert response.status_code == 403


def test_admin_can_access_admin_list_users_endpoint():
    """CRITICAL: Admin user SHOULD be able to list users"""
    # Create admin user directly in database
    db.add_user("admin", hash_password("AdminPass123!"), "productspace@proton.me", RoleEnum.ADMIN)

    # Login as admin
    login_response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "AdminPass123!"}
    )
    token = login_response.json()["access_token"]

    # Access admin endpoint
    response = client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {token}"}
    )

    # MUST be 200 OK
    assert response.status_code == 200
    assert "users" in response.json()
    assert response.json()["count"] >= 1


def test_user_cannot_access_admin_audit_log():
    """User should NOT be able to view audit log"""
    # Create regular user
    signup_response = client.post(
        "/auth/signup",
        json={"username": "alice", "password": "SecurePass123!"}
    )
    token = signup_response.json()["access_token"]

    # Try to access audit log
    response = client.get(
        "/admin/audit-log",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


def test_user_cannot_access_admin_system_stats():
    """User should NOT be able to view system stats"""
    # Create regular user
    signup_response = client.post(
        "/auth/signup",
        json={"username": "alice", "password": "SecurePass123!"}
    )
    token = signup_response.json()["access_token"]

    # Try to access system stats
    response = client.get(
        "/admin/system-stats",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


def test_user_cannot_promote_to_admin():
    """User should NOT be able to promote other users"""
    # Create two regular users
    client.post("/auth/signup", json={"username": "alice", "password": "Pass123!"})
    signup_response = client.post(
        "/auth/signup",
        json={"username": "bob", "password": "Pass123!"}
    )
    token = signup_response.json()["access_token"]

    # Try to promote user
    response = client.post(
        "/admin/promote-to-admin",
        json={"username": "alice"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


def test_admin_can_promote_to_admin():
    """Admin SHOULD be able to promote users"""
    # Create admin and regular user
    db.add_user("admin", hash_password("AdminPass123!"), "productspace@proton.me", RoleEnum.ADMIN)
    db.add_user("alice", hash_password("AlicePass123!"), "productspace@proton.me", RoleEnum.USER)

    # Login as admin
    login_response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "AdminPass123!"}
    )
    token = login_response.json()["access_token"]

    # Promote user
    response = client.post(
        "/admin/promote-to-admin",
        json={"username": "alice"},
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    assert "promoted" in response.json()["message"]


def test_admin_can_delete_user():
    """Admin SHOULD be able to delete users"""
    # Create admin and regular user
    db.add_user("admin", hash_password("AdminPass123!"), "productspace@proton.me", RoleEnum.ADMIN)
    db.add_user("alice", hash_password("AlicePass123!"), "productspace@proton.me", RoleEnum.USER)

    # Login as admin
    login_response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "AdminPass123!"}
    )
    token = login_response.json()["access_token"]

    # Delete user
    response = client.delete(
        "/admin/users/alice",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    assert "deleted" in response.json()["message"]


def test_user_cannot_delete_other_users():
    """User should NOT be able to delete other users"""
    # Create two regular users
    db.add_user("alice", hash_password("AlicePass123!"), "productspace@proton.me", RoleEnum.USER)
    db.add_user("bob", hash_password("BobPass123!"), "productspace@proton.me", RoleEnum.USER)

    # Login as alice
    login_response = client.post(
        "/auth/login",
        json={"username": "alice", "password": "AlicePass123!"}
    )
    token = login_response.json()["access_token"]

    # Try to delete bob
    response = client.delete(
        "/admin/users/bob",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


# ============================================================================
# AUDIT LOGGING TESTS
# ============================================================================

def test_admin_action_creates_audit_log_entry():
    """Admin actions should be logged"""
    # Create admin
    db.add_user("admin", hash_password("AdminPass123!"), "productspace@proton.me", RoleEnum.ADMIN)

    # Login
    login_response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "AdminPass123!"}
    )
    token = login_response.json()["access_token"]

    # Perform admin action
    client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {token}"}
    )

    # Check audit log
    response = client.get(
        "/admin/audit-log",
        headers={"Authorization": f"Bearer {token}"}
    )

    audit_log = response.json()["audit_log"]
    assert len(audit_log) > 0
    assert any(entry["action"] == "LIST_USERS" for entry in audit_log)


def test_user_promotion_audit_logged():
    """User promotion should be logged"""
    db.add_user("admin", hash_password("AdminPass123!"), "productspace@proton.me", RoleEnum.ADMIN)
    db.add_user("alice", hash_password("AlicePass123!"), "productspace@proton.me", RoleEnum.USER)

    login_response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "AdminPass123!"}
    )
    token = login_response.json()["access_token"]

    # Promote user
    client.post(
        "/admin/promote-to-admin",
        json={"username": "alice"},
        headers={"Authorization": f"Bearer {token}"}
    )

    # Check audit log
    response = client.get(
        "/admin/audit-log",
        headers={"Authorization": f"Bearer {token}"}
    )

    audit_log = response.json()["audit_log"]
    assert any(entry["action"] == "PROMOTE_TO_ADMIN" for entry in audit_log)


# ============================================================================
# TOKEN VALIDATION TESTS
# ============================================================================

def test_token_includes_role_claim():
    """Token should include role information"""
    import jwt

    db.add_user("admin", hash_password("AdminPass123!"), "productspace@proton.me", RoleEnum.ADMIN)

    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "AdminPass123!"}
    )
    token = response.json()["access_token"]

    # Decode token (without verification, just to inspect)
    payload = jwt.decode(token, options={"verify_signature": False})

    assert payload["role"] == "admin"
    assert payload["sub"] == "admin"


def test_token_signature_validation():
    """Token with modified payload should be rejected"""
    db.add_user("alice", hash_password("AlicePass123!"), "productspace@proton.me", RoleEnum.USER)

    response = client.post(
        "/auth/login",
        json={"username": "alice", "password": "AlicePass123!"}
    )
    token = response.json()["access_token"]

    # Forge a token by creating a fake one
    forged_token = token[:-10] + "fakesignat"  # Modify signature

    # Try to use forged token
    response = client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {forged_token}"}
    )

    # Should be rejected
    assert response.status_code == 401


# ============================================================================
# PASSWORD SECURITY TESTS
# ============================================================================

def test_password_hashed_with_bcrypt():
    """Passwords should be hashed, not stored plaintext"""
    client.post(
        "/auth/signup",
        json={"username": "alice", "password": "SecurePass123!"}
    )

    stored_hash = db.users["alice"]["password_hash"]

    # Hash should not be plaintext password
    assert stored_hash != "SecurePass123!"
    # Hash should look like bcrypt (starts with $2a$ or $2b$)
    assert stored_hash.startswith("$2")


def test_password_verification_works():
    """Password verification should work correctly"""
    password = "SecurePass123!"
    hashed = hash_password(password)

    # Correct password should verify
    from serving.main_secure import verify_password
    assert verify_password(password, hashed) is True

    # Wrong password should not verify
    assert verify_password("WrongPass123!", hashed) is False


# ============================================================================
# EDGE CASES
# ============================================================================

def test_admin_cannot_delete_self():
    """Admin should not be able to delete their own account"""
    db.add_user("admin", hash_password("AdminPass123!"), "productspace@proton.me", RoleEnum.ADMIN)

    login_response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "AdminPass123!"}
    )
    token = login_response.json()["access_token"]

    # Try to delete self
    response = client.delete(
        "/admin/users/admin",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 400
    assert "Cannot delete your own account" in response.json()["detail"]


def test_metrics_endpoint_admin_only():
    """Metrics endpoint should be admin-only"""
    # Create user
    signup_response = client.post(
        "/auth/signup",
        json={"username": "alice", "password": "SecurePass123!"}
    )
    token = signup_response.json()["access_token"]

    # Try to access metrics as user
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
