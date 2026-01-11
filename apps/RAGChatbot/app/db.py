from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config import DATABASE_URL, SQLA_ECHO, POOL_SIZE, MAX_OVERFLOW, POOL_PRE_PING

class Base(DeclarativeBase):
    """全モデルの親:DDL(設計図)の台帳を提供する"""
    pass

# エンジンの作成
engine = create_engine(
    DATABASE_URL,
    echo=SQLA_ECHO,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_pre_ping=POOL_PRE_PING,
    future=True,
)

# リクエスト単位で使う Session作る工場
SessionLocal = sessionmaker(
    bind = engine,
    autoflush=False,
    expire_on_commit=False,
    future=True
)
