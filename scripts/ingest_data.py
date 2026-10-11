import os
import json
import sqlite3
import requests
import chromadb
from chromadb.config import Settings

# --- Paths & Configuration ---
RAW_JSON_PATH = "data/raw/india_tourism_dataset.json"
PROCESSED_DIR = "data/processed"
SQLITE_DB_PATH = os.path.join(PROCESSED_DIR, "tourism.db")
CHROMA_DIR = os.path.join(PROCESSED_DIR, "chroma_db")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"


def get_ollama_embedding(text: str) -> list[float]:
    """Fetch vector embeddings directly from local Ollama server."""
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "prompt": text}
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama embedding request failed: {response.text}")
    return response.json()["embedding"]


def setup_sqlite(data: list[dict]):
    """Ingest dataset into SQLite table with flattened fields."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Drop existing table to ensure a clean build
    cursor.execute("DROP TABLE IF EXISTS destinations")

    cursor.execute("""
    CREATE TABLE destinations (
        id INTEGER PRIMARY KEY,
        destination_name TEXT,
        state TEXT,
        district TEXT,
        region TEXT,
        
        -- Flattened Daily Budget Ranges (INR)
        budget_daily_min INT,
        budget_daily_max INT,
        mid_range_daily_min INT,
        mid_range_daily_max INT,
        luxury_daily_min INT,
        luxury_daily_max INT,
        
        -- Duration & Safety
        minimum_days INT,
        ideal_days INT,
        maximum_days INT,
        safety_rating INT,
        
        -- JSON string arrays for matching
        trip_types TEXT,
        ideal_for TEXT,
        best_seasons TEXT,
        primary_attractions TEXT
    )
    """)

    for row in data:
        budget_range = row.get("budget_category", {}).get("total_daily_range", [0, 0])
        mid_range = row.get("mid_range_category", {}).get("total_daily_range", [0, 0])
        luxury_range = row.get("luxury_category", {}).get("total_daily_range", [0, 0])

        cursor.execute("""
        INSERT INTO destinations (
            id, destination_name, state, district, region,
            budget_daily_min, budget_daily_max,
            mid_range_daily_min, mid_range_daily_max,
            luxury_daily_min, luxury_daily_max,
            minimum_days, ideal_days, maximum_days, safety_rating,
            trip_types, ideal_for, best_seasons, primary_attractions
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("id"),
            row.get("destination_name", row.get("state")),
            row.get("state"),
            row.get("district"),
            row.get("region"),
            
            budget_range[0] if len(budget_range) > 0 else 0,
            budget_range[1] if len(budget_range) > 1 else 0,
            mid_range[0] if len(mid_range) > 0 else 0,
            mid_range[1] if len(mid_range) > 1 else 0,
            luxury_range[0] if len(luxury_range) > 0 else 0,
            luxury_range[1] if len(luxury_range) > 1 else 0,
            
            row.get("minimum_days", 1),
            row.get("ideal_days", 3),
            row.get("maximum_days", 7),
            row.get("safety_rating", 5),
            
            json.dumps([t.lower() for t in row.get("trip_types", [])]),
            json.dumps([i.lower() for i in row.get("ideal_for", [])]),
            json.dumps([s.lower() for s in row.get("best_seasons", [])]),
            json.dumps(row.get("primary_attractions", []))
        ))

    conn.commit()
    conn.close()
    print(f"✅ SQLite Ingestion Complete: {len(data)} destinations stored at '{SQLITE_DB_PATH}'.")


def setup_chromadb(data: list[dict]):
    """Embed rich descriptive text into ChromaDB for semantic search."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        client.delete_collection("india_tourism")
    except Exception:
        pass

    collection = client.create_collection(
        name="india_tourism",
        metadata={"hnsw:space": "cosine"}
    )

    print("⏳ Generating vector embeddings via local Ollama nomic-embed-text...")

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for index, row in enumerate(data):
        dest_name = row.get("destination_name", row.get("state"))
        state = row.get("state", "")
        experiences = row.get("unique_experiences", "")
        attractions = ", ".join(row.get("primary_attractions", []))
        hidden_gems = ", ".join(row.get("hidden_gems", []))
        food_scene = row.get("food_scene", "")
        itinerary = row.get("suggested_itinerary", "")
        user_reviews = row.get("user_reviews_summary", "")

        text_passage = (
            f"Destination: {dest_name}, State: {state}. "
            f"Unique Experiences: {experiences}. "
            f"Key Attractions: {attractions}. "
            f"Hidden Gems: {hidden_gems}. "
            f"Food & Cuisine: {food_scene}. "
            f"Suggested Itinerary Overview: {itinerary}. "
            f"Reviews Summary: {user_reviews}."
        )

        vector = get_ollama_embedding(text_passage)

        documents.append(text_passage)
        embeddings.append(vector)
        metadatas.append({
            "dest_id": row.get("id", index + 1),
            "destination_name": dest_name,
            "state": state,
            "region": row.get("region", "")
        })
        ids.append(str(row.get("id", index + 1)))

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ ChromaDB Vector Ingestion Complete: {len(documents)} vectors saved at '{CHROMA_DIR}'.")


if __name__ == "__main__":
    if not os.path.exists(RAW_JSON_PATH):
        raise FileNotFoundError(f"Missing dataset file at '{RAW_JSON_PATH}'. Ensure file is placed correctly.")

    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        tourism_data = json.load(f)

    if isinstance(tourism_data, dict):
        tourism_data = tourism_data.get("destinations", [tourism_data])

    print(f"Starting ingestion for {len(tourism_data)} dataset entries...")
    setup_sqlite(tourism_data)
    setup_chromadb(tourism_data)