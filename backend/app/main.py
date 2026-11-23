import loguru
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import users, admin
from app.api.v1 import routes as v1_routes
from app.api.auth import auth_backend, fastapi_users
from app.schemas import UserRead, UserCreate
from app.core.db import async_engine
from app.models.base import Base
import app.models  # noqa: F401 — register every model on Base.metadata before create_all


app = FastAPI()

# CORS Middleware - Restrict origins in production
allowed_origins = ["http://localhost:5173", "http://localhost:80", "http://localhost:8090"]
if os.getenv("ENVIRONMENT") == "development":
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Authentication routes (provided by fastapi-users)
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/api/v1/auth/jwt",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/api/v1/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/api/v1/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/api/v1/auth",
    tags=["auth"],
)

# Application routes
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(admin.router, prefix="/api", tags=["admin"])  # admin router carries its own /admin prefix
app.include_router(v1_routes.router, prefix="/api/v1", tags=["analysis"])

# Standalone file upload (router carries its own /upload prefix)
from app.api.v1 import upload as v1_upload
app.include_router(v1_upload.router, prefix="/api/v1")

# Upload + AI detection in one call (router carries its own /upload-ai prefix)
from app.api.v1 import upload_ai as v1_upload_ai
app.include_router(v1_upload_ai.router, prefix="/api/v1")

# Moodle server-to-server integration (X-API-Key auth)
from app.api.v1 import moodle as v1_moodle
app.include_router(v1_moodle.router, prefix="/api/v1", tags=["moodle"])


@app.on_event("startup")
async def startup_event():
    loguru.logger.info("Starting up...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Apply additive column changes create_all cannot make on existing tables.
    from app.core.schema_guard import ensure_schema
    await ensure_schema(async_engine)

    # Seed the database with initial data
    try:
        from app.core.database_seed import seed_database
        await seed_database()
    except Exception as e:
        loguru.logger.error(f"Database seeding failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    loguru.logger.info("Shutting down...")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Plagiarism Detection API", "status": "running"}
