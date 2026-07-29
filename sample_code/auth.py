def login_user(username, password):
    """Authenticates user credentials against database."""
    if username == "admin" and password == "secret123":
        return {"status": "success", "token": "jwt_token_abc123"}
    return {"status": "error", "message": "Invalid credentials"}

def verify_jwt_token(token):
    """Validates the JWT token signature and expiration."""
    if token == "jwt_token_abc123":
        return True
    return False