from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTStrategy, AuthenticationBackend, BearerTransport
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users.exceptions import UserAlreadyExists
from fastapi_users.manager import BaseUserManager, UUIDIDMixin
from app.models.user import User
from app.core.db import AsyncSessionLocal, get_db
from app.schemas import UserCreate, UserRead, UserUpdate
from fastapi_users import schemas as fastapi_users_schemas
from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
from uuid import UUID
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Bearer transport: login returns {access_token, token_type} in the response body,
# which the frontend stores and sends as an Authorization: Bearer header.
# (CookieTransport with cookie_secure=True broke browser login over plain HTTP.)
bearer_transport = BearerTransport(tokenUrl="api/v1/auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=settings.SECRET_KEY, lifetime_seconds=3600)


# Authentication backend
auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)


# User manager and FastAPI Users setup
class UserManager(UUIDIDMixin, BaseUserManager[User, UUID]):
    def __init__(self, user_db: SQLAlchemyUserDatabase):
        super().__init__(user_db)

    async def create(self, user_create, safe: bool = False, request=None):
        """
        Guard privilege escalation: `role` is a custom field fastapi-users' `safe`
        flag doesn't know about, so a self-registrant could POST role=admin.
        On the self-service path (safe=True) force role to 'user'; only an admin
        calling with safe=False may set a privileged role.
        """
        if safe and getattr(user_create, "role", None) not in (None, "user"):
            user_create = user_create.copy(update={"role": "user"})
        return await super().create(user_create, safe=safe, request=request)

    async def on_after_register(self, user: User, request=None):
        print(f"User {user.id} has registered.")
        
    async def on_after_forgot_password(self, user: User, token: str, request=None):
        print(f"User {user.id} has forgot their password. Reset token: {token}")
        
    async def on_after_request_verify(self, user: User, token: str, request=None):
        print(f"Verification requested for user {user.id}. Verification token: {token}")


async def get_user_db(session: AsyncSession = Depends(get_db)):
    yield SQLAlchemyUserDatabase(session, User)


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)


# Initialize FastAPI Users
fastapi_users = FastAPIUsers[User, UUID](
    get_user_manager,
    [auth_backend],
)


# Current active user dependency
current_user = fastapi_users.current_user(active=True)


# Admin dependency
async def admin_user(user: User = Depends(current_user)):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Admin privileges required."
        )
    return user


# Moderator or Admin dependency
async def mod_or_admin_user(user: User = Depends(current_user)):
    if user.role not in ["admin", "moderator"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Admin or moderator privileges required."
        )
    return user