from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "mysql+pymysql://root:Lavanya12311@localhost/bullbear_ai"

# Create engine 
engine = create_engine(
    DATABASE_URL,
    echo=True,        # Shows SQL queries (good for learning)
    future=True
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models  ✅ THIS WAS MISSING
Base = declarative_base()
