from datetime import datetime
from . import db 
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase,Mapped,mapped_column
import datetime as dt
class Base(DeclarativeBase):
    pass
class Conversation(db.Model,Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(db.integer,primary_key=True)
    created_at: Mapped[dt.datetime] =mapped_column( db.Column(db.DateTime, default=datetime.utcnow))


class Message(db.Model,Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(db.integer, primary_key=True)
    conversation_id: Mapped[int] =mapped_column(db.integer,db.ForeignKey,nullable=False)
    role: Mapped[int] = mapped_column(db.String(20), nullable=False)
    content: Mapped[str] = mapped_column(db.String,nullable=False)
    created_at: Mapped[dt.datetime] =mapped_column( db.Column(db.DateTime, default=datetime.utcnow))
