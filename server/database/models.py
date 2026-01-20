from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Portfolio(Base):
    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True, index=True)

    # User & stock identity
    user_id = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    # Investment details
    quantity = Column(Float, default=0.0)
    buy_price = Column(Float, default=0.0)

    # Portfolio analytics (IMPORTANT)
    weight = Column(Float, default=0.0)   # ✅ FIXES YOUR ERROR

    created_at = Column(DateTime, default=datetime.utcnow)
