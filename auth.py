from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from sqlmodel import select
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer
from db import get_session
from models import Admin
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# تنظیمات امنیت
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# برای دریافت توکن در مسیرهای محافظت‌شده
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login")

# --------- Models ----------
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

# --------- Helpers ----------
def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_admin(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="توکن نامعتبر است.")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="توکن نامعتبر است.")

# --------- Routes ----------
@router.post("/admin/login", response_model=TokenResponse)
def login_admin(data: LoginRequest):
    session = get_session()
    user = session.exec(select(Admin).where(Admin.email == data.email)).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="ایمیل یا رمز عبور نادرست است")

    token = create_access_token(data={"sub": user.email})
    return {"access_token": token}
