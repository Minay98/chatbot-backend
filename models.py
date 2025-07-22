from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy import Column, Text  # ✅ اضافه شده

class ChatHistory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str = Field(sa_column=Column(Text))  # ✅ فقط از sa_column استفاده کن
    answer: str = Field(sa_column=Column(Text))    # ✅ این خط باعث خطا شده بود
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SessionLimit(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str
    ip: str
    message_count: int = 0
    last_message_time: datetime

class UploadedFile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    filepath: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

class Admin(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
