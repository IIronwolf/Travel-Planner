import os

# --- Model Settings ---
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama2"
EMBEDDING_MODEL = "nomic-embed-text"

# --- Dataset Paths ---
RAW_JSON_PATH = "data/raw/india_tourism_dataset.json"
SQLITE_DB_PATH = "data/processed/tourism.db"
CHROMA_DIR = "data/processed/chroma_db"