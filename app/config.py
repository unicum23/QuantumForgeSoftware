import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    index_path: str = os.getenv("INDEX_PATH", "artifacts/index/faiss.index")
    chunks_path: str = os.getenv("CHUNKS_PATH", "artifacts/index/chunks.jsonl")
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.35"))
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")


settings = Settings()
