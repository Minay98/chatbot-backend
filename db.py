from sqlmodel import create_engine, SQLModel, Session

# تنظیمات اتصال به MySQL 8
DATABASE_URL = "mysql+pymysql://mina:mina@192.168.90.21:3306/chatbot?charset=utf8mb4"

# ساخت Engine
engine = create_engine(
    DATABASE_URL,
    echo=True,  # برای مشاهده queryها در لاگ
    pool_pre_ping=True  # جلوگیری از خطای connection timeout
)

# تابع ساخت جدول‌ها (فقط وقتی لازم بود اجرا کن)
def create_db_and_tables():
    print("🛠 Creating tables in DB...")
    SQLModel.metadata.create_all(engine)
    print("✅ Tables created.")

# ساخت Session
def get_session():
    return Session(engine)
