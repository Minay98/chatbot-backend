from fastapi import FastAPI, File, Path, UploadFile, HTTPException,Depends
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from models import ChatHistory, SessionLimit, UploadedFile
from sqlalchemy import delete
from datetime import datetime
import logging
from typing import List
import os
import uuid
from langchain.document_loaders import TextLoader
from langchain.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,UnstructuredWordDocumentLoader
)
from auth import router as auth_router
import chromadb
from models import UploadedFile
from fastapi import Request
from fastapi import FastAPI
from db import engine 
from db import get_session
from db import create_db_and_tables
from langchain_core.documents import Document  # اگر قبلاً import نشده
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from models import Admin
from auth import router as auth_router
from auth import get_password_hash, get_current_admin
UPLOAD_DIR = "./uploaded_files"
chromadb.config.Settings(anonymized_telemetry=False)


app = FastAPI()

app.include_router(auth_router)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
create_db_and_tables()

def create_admin_user():
    with get_session() as session:
        exists = session.exec(select(Admin).where(Admin.email == "admin@example.com")).first()
        if not exists:
            admin = Admin(email="admin@example.com", password_hash=get_password_hash("123456"))
            session.add(admin)
            session.commit()

create_admin_user()

# تست مسیر محافظت‌شده
@app.get("/admin/me")
def read_admin(current_admin: str = Depends(get_current_admin)):
    return {"email": current_admin}


class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    
    
def rebuild_chroma():
    logger.info("📦 بازسازی Chroma vector store پس از حذف")

    embedding = OllamaEmbeddings(
        model="dengcao/Qwen3-Embedding-4B:Q8_0",
        base_url="http://192.168.90.20:7869/"
    )

    session = get_session()
    files = session.exec(select(UploadedFile)).all()

    all_docs = []
    for file in files:
        try:
            file_path = file.filepath
            ext = os.path.splitext(file.filename)[-1].lower()
            if not os.path.exists(file_path):
                continue

            if ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif ext in [".xls", ".xlsx"]:
                loader = UnstructuredExcelLoader(file_path)
            elif ext in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif ext == ".csv":
                loader = CSVLoader(file_path, encoding="utf-8")
            else:
                loader = TextLoader(file_path, encoding="utf-8")

            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
            split_docs = splitter.split_documents(documents)

            docs_with_meta = [
                Document(page_content=doc.page_content, metadata={"filename": file.filename})
                for doc in split_docs
            ]
            all_docs.extend(docs_with_meta)
        except Exception as e:
            logger.warning(f"❌ خطا در بازسازی فایل {file.filename}: {e}")

    # حذف دیتابیس قبلی
    import shutil
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    # بازسازی کامل
    Chroma.from_documents(
        documents=all_docs,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    

    logger.info("✅ بردارها با موفقیت بازسازی شدند.")


def add_history(question: str, answer: str, session_id: str):
    with Session(engine) as session:
        session.add(ChatHistory(question=question, answer=answer, session_id=session_id))
        session.commit()
# Load document function
def load_createDoc(doc_path="doc.txt"):
    try:
        loader = TextLoader(doc_path, encoding="utf-8")
        documents = loader.load()
        if not documents:
            raise ValueError("فایل داده‌ای ندارد یا پردازش نشده است.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        return docs
    except Exception as e:
        logger.error(f"❌ خطا در بارگذاری وکتورها: {e}")
        raise
# Load document function
def load_vectorstore(docs):
    try:
        embedding = OllamaEmbeddings(
            model="dengcao/Qwen3-Embedding-4B:Q8_0",
            base_url="http://192.168.90.20:7869/"
        )
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory="./chroma_db"
        )

        return vectordb.as_retriever()
    except Exception as e:
        logger.error(f"❌ خطا در بارگذاری وکتورها: {e}")
        raise
    
    
    

llm = OllamaLLM(model="gemma3n:latest", base_url="http://192.168.90.20:7869/", name="mina ai")
prompt = PromptTemplate(
    template="""
شما یک دستیار هوش مصنوعی حرفه‌ای هستید که پاسخ‌ها را به زبان فارسی ساده و دقیق می‌دهید.
اگر اطلاعاتی نداشتی یا نمیدونستی فقط بگو که اطلاعاتی ندارم. همچنین اگر 
لطفاً بر اساس سوال زیر و اطلاعات بازیابی شده پاسخ بده:
سوال: {question}
=========
اطلاعات:
{context}
=========
پاسخ:""",
    input_variables=["question", "context"]
)

class Question(BaseModel):
    question: str
    
    
def get_qa_chain():
    embedding = OllamaEmbeddings(
        model="dengcao/Qwen3-Embedding-4B:Q8_0",
        base_url="http://192.168.90.20:7869/"
    )
    retriever1 = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    ).as_retriever()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever1,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    

@app.get("/get_session_id")
def get_session_id():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}
#########################################################################      
class AskRequest(BaseModel):
    question: str
    session_id: str
    
@app.post("/ask")
def ask(req: AskRequest, request: Request):
    try:
        session_id = req.session_id
        question = req.question
        user_ip = request.client.host  

        print("✅ session_id:", session_id)
        print("✅ user_ip:", user_ip)

        with Session(engine) as session:
            # بررسی محدودیت بر اساس session_id

            limit = session.exec(
    select(SessionLimit).where(SessionLimit.session_id == session_id)
).first()

            if not limit:
                limit = SessionLimit(
                    session_id=session_id,
                    ip=user_ip,
                    message_count=0,
                    last_message_time=datetime.utcnow()
                )
                session.add(limit)
                session.commit()
                session.refresh(limit)
       

            if limit.message_count >= 50:
                raise HTTPException(status_code=403, detail="به سقف مجاز پیام رسیده‌اید.")

            # اجرای مدل
            qa_chain = get_qa_chain()
            result = qa_chain.invoke({"query": question})
            answer = result.get("result", "")
            sources = result.get("source_documents", [])

            if not sources:
                answer = "❌ پاسخ این سؤال در اطلاعات من موجود نیست یا مرتبط با حوزه کاری من نمی‌باشد."

            # ذخیره تاریخچه
            add_history(question=question, answer=answer, session_id=session_id)
            limit.message_count += 1
            session.commit()

        return {"question": question, "answer": answer}

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        logger.error(f"❌ خطا در /ask: {e}")
        raise HTTPException(status_code=500, detail="خطای داخلی در پردازش مدل")
#########################################################################      
@app.get("/history")
def get_history():
    with Session(engine) as session:
        chats = session.exec(select(ChatHistory).order_by(ChatHistory.timestamp.desc())).all()
        return [
            {
                "question": chat.question,
                "answer": chat.answer,
                "timestamp": chat.timestamp.strftime("%Y-%m-%d %H:%M") if chat.timestamp else None
            }
            for chat in chats
        ]
#########################################################################      
@app.delete("/history")
def delete_history(session_id: str):
    with Session(engine) as session:
        session.exec(
            delete(ChatHistory).where(ChatHistory.session_id == session_id)
        )
        session.commit()
    return {"message": "تاریخچه گفتگو حذف شد."}
#########################################################################      

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        all_docs = []
        uploaded_files = []
        skipped_files = []

        embedding = OllamaEmbeddings(
            model="dengcao/Qwen3-Embedding-4B:Q8_0",
            base_url="http://192.168.90.20:7869/"
        )

        session = get_session()

        if os.path.exists("./chroma_db"):
            vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
        else:
            vectordb = Chroma.from_documents([], embedding=embedding, persist_directory="./chroma_db")

        for file in files:
            try:
                # بررسی تکراری بودن
                existing_file = session.exec(
                    select(UploadedFile).where(UploadedFile.filename == file.filename)
                ).first()
                if existing_file:
                    logger.warning(f"⚠️ فایل تکراری رد شد: {file.filename}")
                    skipped_files.append(file.filename)
                    continue

                content = await file.read()
                if not content:
                    raise HTTPException(status_code=400, detail=f"❌ فایل {file.filename} خالی است.")

                file_path = os.path.join(UPLOAD_DIR, file.filename)
                with open(file_path, "wb") as f:
                    f.write(content)

                ext = os.path.splitext(file.filename)[-1].lower()
                if ext not in [".txt", ".pdf", ".xls", ".xlsx", ".doc", ".docx", ".csv"]:
                    logger.warning(f"📂 فرمت فایل {file.filename} پشتیبانی نمی‌شود. فقط ذخیره شد.")
                    uploaded_files.append(file.filename)
                    continue

                if ext == ".pdf":
                    loader = PyMuPDFLoader(file_path)
                elif ext in [".xls", ".xlsx"]:
                    loader = UnstructuredExcelLoader(file_path)
                elif ext in [".doc", ".docx"]:
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif ext == ".csv":
                    loader = CSVLoader(file_path, encoding="utf-8")
                else:
                    loader = TextLoader(file_path, encoding="utf-8")

                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
                split_docs = splitter.split_documents(documents)

                # تست اولیه embed
                test_embed = embedding.embed_query(split_docs[0].page_content)
                if not test_embed or len(test_embed) == 0:
                    logger.warning(f"🚫 مدل embedding پاسخ نداد برای فایل: {file.filename}")
                    skipped_files.append(file.filename)
                    os.remove(file_path)  # حذف فایل از فایل‌سیستم
                    continue  # رد کردن این فایل

                docs_with_metadata = [
                    Document(page_content=doc.page_content, metadata={"filename": file.filename})
                    for doc in split_docs
                ]

                # ثبت در دیتابیس (بعد از تأیید سالم بودن embedding)
                file_record = UploadedFile(filename=file.filename, filepath=file_path)
                session.add(file_record)

                vectordb.add_documents(docs_with_metadata)

                logger.info(f"✅ فایل: {file.filename} - بخش‌ها: {len(split_docs)}")
                all_docs.extend(split_docs)
                uploaded_files.append(file.filename)

            except UnicodeDecodeError as ude:
                logger.error(f"❌ خطای encoding در فایل {file.filename}: {ude}")
                raise HTTPException(status_code=400, detail=f"خطای encoding در فایل {file.filename}")
            except Exception as e:
                logger.error(f"❌ خطا در پردازش فایل {file.filename}: {e}")
                raise HTTPException(status_code=400, detail=f"خطا در پردازش فایل {file.filename}")

        session.commit()
        session.close()

        return {
            "message": "فرآیند آپلود کامل شد.",
            "uploaded_files": uploaded_files,
            "skipped_duplicates": skipped_files,
            "total_docs_added": len(all_docs)
        }

    except Exception as e:
        logger.error(f"❌ خطا در بارگذاری فایل‌ها: {e}")
        raise HTTPException(status_code=500, detail="خطای داخلی در هنگام بارگذاری فایل‌ها")


#########################################################################          
@app.get("/files")
def list_uploaded_files():
    directory = UPLOAD_DIR  # مسیر فایل‌ها، مثلاً /uploads یا ./files
    files = []
    for filename in os.listdir(directory):
        files.append(filename)
    return {"files": files}
#########################################################################          
@app.delete("/files/{filename}")
def delete_file(filename: str = Path(...)):
    file_path = os.path.join(UPLOAD_DIR, filename)
    session = get_session()

    if os.path.exists(file_path):
        os.remove(file_path)

    # حذف از دیتابیس
    file_record = session.exec(
        select(UploadedFile).where(UploadedFile.filename == filename)
    ).first()

    if file_record:
        session.delete(file_record)
        session.commit()

    # حذف از Chroma
    try:
        embedding = OllamaEmbeddings(
            model="dengcao/Qwen3-Embedding-4B:Q8_0",
            base_url="http://192.168.90.20:7869/"
        )
        vectordb = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding
        )

        vectordb._collection.delete(where={"filename": filename})  # ✅ حذف دقیق
        vectordb.persist()  # ✅ ذخیره تغییرات

        logger.info(f"✅ اسناد مرتبط با {filename} از بردار حذف شدند.")

    except Exception as e:
        logger.error(f"❌ خطا در حذف اسناد از ChromaDB: {e}")

    return {"message": f"{filename} از همه‌جا حذف شد."}


#########################################################################      
@app.get("/chat/{session_id}")
def get_chat(session_id: str):
    with Session(engine) as session:
        history = session.exec(
            select(ChatHistory).where(ChatHistory.session_id == session_id)
        ).all()
        return [
            {
                "question": msg.question,
                "answer": msg.answer,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in history
        ]
        
####################################################################
