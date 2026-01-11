import datetime as dt
from sqlalchemy.orm import Mapped, mapped_column
from .extensions import db

class Conversation(db.Model):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(db.Integer, primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(db.DateTime, default=dt.datetime.utcnow)


class Message(db.Model):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(db.Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        db.Integer, db.ForeignKey("conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(db.String(20), nullable=False)
    content: Mapped[str] = mapped_column(db.Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(db.DateTime, default=dt.datetime.utcnow)
