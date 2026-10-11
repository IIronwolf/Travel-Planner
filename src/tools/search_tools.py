import json
import sqlite3
import requests
import chromadb
from typing import List, Dict, Any
from src.config import SQLITE_DB_PATH, CHROMA_DIR

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"

def get_ollama_embedding(text: str) -> List[float]:
    """Generate vector embedding using local Ollama."""
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "prompt": text}
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama embedding request failed: {response.text}")
    return response.json()["embedding"]


def sql_search(query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Query SQLite for hard categorical, geographic, and budget constraints."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    sql = "SELECT * FROM destinations WHERE 1=1"
    args = []

    dest_name = query_params.get("destination_name")
    if dest_name:
        sql += " AND (LOWER(destination_name) LIKE ? OR LOWER(state) LIKE ? OR LOWER(district) LIKE ?)"
        pattern = f"%{dest_name.lower()}%"
        args.extend([pattern, pattern, pattern])

    region = query_params.get("region")
    if region:
        sql += " AND LOWER(region) = ?"
        args.append(region.lower())

    duration = query_params.get("duration_days")
    if duration:
        sql += " AND minimum_days <= ? AND maximum_days >= ?"
        args.extend([duration, duration])

    max_budget = query_params.get("max_daily_budget_inr")
    budget_tier = query_params.get("budget_tier", "budget")
    if max_budget:
        if budget_tier == "mid_range":
            sql += " AND mid_range_daily_min <= ?"
        elif budget_tier == "luxury":
            sql += " AND luxury_daily_min <= ?"
        else:
            sql += " AND budget_daily_min <= ?"
        args.append(max_budget)

    cursor.execute(sql, args)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    filtered_results = []
    for r in rows:
        is_match = True

        req_types = query_params.get("trip_types", [])
        if req_types:
            db_types = json.loads(r["trip_types"]) if r["trip_types"] else []
            if not any(t.lower() in [d.lower() for d in db_types] for t in req_types):
                is_match = False

        ideal = query_params.get("ideal_for")
        if ideal and is_match:
            db_ideal = json.loads(r["ideal_for"]) if r["ideal_for"] else []
            if ideal.lower() not in [i.lower() for i in db_ideal]:
                is_match = False

        season = query_params.get("season")
        if season and is_match:
            db_seasons = json.loads(r["best_seasons"]) if r["best_seasons"] else []
            if season.lower() not in [s.lower() for s in db_seasons]:
                is_match = False

        if is_match:
            filtered_results.append(r)

    return filtered_results

def vector_search(vibe_description: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query ChromaDB for semantic similarity on vibes and experiences."""
    if not vibe_description:
        return []

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name="india_tourism")

    query_vec = get_ollama_embedding(vibe_description)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k
    )

    matches = []
    if results and results.get("ids"):
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [0] * len(ids)

        for dest_id, meta, dist in zip(ids, metadatas, distances):
            matches.append({
                "id": int(dest_id),
                "destination_name": meta.get("destination_name"),
                "state": meta.get("state"),
                "region": meta.get("region"),
                "similarity_score": round(1 - dist, 4) if dist <= 1 else round(dist, 4)
            })

    return matches

def hybrid_search(query_params: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """Combines exact SQL criteria with semantic vector similarity scoring."""
    sql_results = sql_search(query_params)
    sql_map = {r["id"]: r for r in sql_results}

    vibe = query_params.get("vibe_description")
    vector_results = vector_search(vibe, top_k=top_k) if vibe else []

    combined = []
    seen_ids = set()

    for vec_item in vector_results:
        dest_id = vec_item["id"]
        seen_ids.add(dest_id)
        if dest_id in sql_map:
            record = sql_map[dest_id]
            record["match_type"] = "exact_and_vibe"
            record["score"] = vec_item.get("similarity_score", 1.0)
            combined.append(record)

    for dest_id, record in sql_map.items():
        if dest_id not in seen_ids:
            record["match_type"] = "constraint_only"
            record["score"] = 0.5
            combined.append(record)

    if not combined and vector_results:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        for vec_item in vector_results:
            cursor.execute("SELECT * FROM destinations WHERE id = ?", (vec_item["id"],))
            row = cursor.fetchone()
            if row:
                rec = dict(row)
                rec["match_type"] = "vibe_only"
                rec["score"] = vec_item.get("similarity_score", 0.5)
                combined.append(rec)
        conn.close()

    return combined[:top_k]