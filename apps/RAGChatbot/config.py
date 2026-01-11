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

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")

def _env_list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if val is None:
        return default
    parts = [p.strip() for p in val.split(",")]
    return [p for p in parts if p]

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
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLA_ECHO = _env_bool("SQLA_ECHO", False)
    POOL_SIZE = int(os.getenv("SQLA_POOL_SIZE", "5"))
    MAX_OVERFLOW = int(os.getenv("SQLA_MAX_OVERFLOW", "10"))
    POOL_PRE_PING = _env_bool("SQLA_POOL_PRE_PING", True)
    SCOPE_KEYWORDS = _env_list(
        "SCOPE_KEYWORDS",
        ["補助金", "助成金", "給付金", "支援制度", "支援金", "助成制度"],
    )
