# Flask Session・PostgreSQL 永続化・Alembic マイグレーション導入ガイド

## 📘 概要
RAG チャットアプリに、以下の基盤機能を導入するための手順とサンプルコードをまとめた。

- Flask セッション管理  
- PostgreSQL16（Docker コンテナ）  
- SQLAlchemy モデル  
- Alembic（Flask-Migrate）によるマイグレーション管理  

これにより、**会話単位の `conversation_id` をセッションで記録し、メッセージ履歴を DB に永続化できる構成**となる。

---

# 1. ディレクトリ構成（例）

```
project/
  docker-compose.yml
  requirements.txt
  app/
    __init__.py
    models.py
```

---

# 2. Docker（PostgreSQL16）

`docker-compose.yml`

```yaml
version: "3.9"

services:
  db:
    image: postgres:16
    container_name: rag-postgres
    environment:
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ragpass
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - db-data:/var/lib/postgresql/data

volumes:
  db-data:
```

---

# 3. Python パッケージ

```
Flask
flask_sqlalchemy
flask_migrate
psycopg2-binary
```

---

# 4. Flask アプリ（セッション & DB）

`__init__.py`

```python
import os
from flask import Flask, session, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()


def create_app():
    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://raguser:ragpass@db:5432/ragdb",
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    migrate.init_app(app, db)

    from .models import Conversation, Message  # noqa

    def start_conversation():
        conv = Conversation()
        db.session.add(conv)
        db.session.commit()
        session["conversation_id"] = conv.id
        return conv.id

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json()
        user_text = data["text"]

        conv_id = session.get("conversation_id")
        if conv_id is None:
            conv_id = start_conversation()

        user_msg = Message(
            conversation_id=conv_id,
            role="user",
            content=user_text,
        )
        db.session.add(user_msg)

        answer_text = f"ダミー応答: {user_text}"

        assistant_msg = Message(
            conversation_id=conv_id,
            role="assistant",
            content=answer_text,
        )
        db.session.add(assistant_msg)
        db.session.commit()

        return jsonify({"answer": answer_text, "conversation_id": conv_id})

    @app.route("/api/reset", methods=["POST"])
    def reset():
        session.pop("conversation_id", None)
        return "", 204

    return app
```

---

# 5. モデル

`models.py`

```python
from datetime import datetime
from . import db


class Conversation(db.Model):
    __tablename__ = "conversations"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Message(db.Model):
    __tablename__ = "messages"
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(
        db.Integer,
        db.ForeignKey("conversations.id"),
        nullable=False,
    )
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

---

# 6. Alembic（Flask-Migrate）

```
$env:FLASK_APP = "app:create_app"
$env:FLASK_ENV = "development"
flask db init
flask db migrate -m "initial tables"
flask db upgrade
```

---

# 7. ブランチ例

```
git checkout -b feat/3-session-db-migration
```

---

# ✔ 完了
この Markdown はダウンロードして GitHub に貼る・内部資料として共有できる形式になっています。
