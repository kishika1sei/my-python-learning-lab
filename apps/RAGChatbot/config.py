# config.py
import os

# DB接続URLを作成する関数
def _buils_database_url_from_parts() -> str:
    user = os.getenv("POSTGRES_USER", "appuser")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "db")
    port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "appdb")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

# 接続URLはDATABASE_URLを優先し、なければ POSTGRE_* から組み立てる
class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    PDF_DIR = os.getenv("PDF_DIR", "data/pdf")
    INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
    CTX_MAX_CHUNKS = 4
    CTX_MAX_CHARS = 1500
    SYS_PROMPT = os.getenv("SYS_PROMPT", "あなたは日本語で正確に答えるアシスタントです。根拠に基づき簡潔に回答し、不明な点は正直に『不明』と述べてください。")
    DATABASE_URL = os.getenv("DATABASE_URL") or _buils_database_url_from_parts()
