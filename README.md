# ✈️ Offline Indian Travel AI Agent

An intelligent, fully offline travel concierge designed to parse natural language queries, execute hybrid search (categorical SQL filtering + semantic vector matching), and generate structured day-by-day travel itineraries across India.

Powered by **Google ADK**, **Ollama (Llama 2 & nomic-embed-text)**, **SQLite** and **ChromaDB**.

---

## 🏛️ Architecture & System Flow

+------------------+      +---------------------+      +---------------------+
| User Query (Text)| ---> | Orchestrator Agent  | ---> | TravelQuerySchema   |
+------------------+      | (Google ADK / Llama2|      | (Pydantic JSON)     |
+---------------------+      +----------+----------+
|
v
+------------------+      +---------------------+      +----------+----------+
|  Presenter Agent | <--- |   Hybrid Retrieval  | <--- | SQL & Vector Search |
| (Markdown Output)|      | (Merges Constraints)|      | (SQLite + ChromaDB) |
+------------------+      +---------------------+      +---------------------+

* **Orchestrator Agent**: Powered by Google ADK (`google-adk`) and local `llama2` via LiteLLM. Extracts structured travel parameters (destination, region, trip duration, budget tier, maximum daily budget, season, group archetype, and vibe keywords) into a validated `TravelQuerySchema`.
* **Hybrid Retrieval Engine**:
  * **SQLite (`tourism.db`)**: Performs hard filtering on numeric ranges (daily budget min/max, duration limits) and exact categorical tags (region, traveler archetype, seasons).
  * **ChromaDB (`chroma_db`)**: Runs dense vector similarity search using `nomic-embed-text` embeddings for subjective preferences (e.g., quiet beaches, Portuguese architecture, local seafood, hidden gems).
  * **Hybrid Ranker**: Merges exact SQL constraints with semantic similarity scores to return top destination matches.
* **Presenter Agent**: Formats dataset records into a personalized, daily travel itinerary.
* **100% Offline Capable**: Runs locally without external cloud API dependencies.

---

## 🛠️ Tech Stack

* **Agent Framework**: Google ADK (`google-adk`), LiteLLM (`litellm`)
* **Local Inference Engine**: Ollama (`llama2` & `nomic-embed-text`)
* **Databases**: SQLite3 (Metadata/Filtering) & ChromaDB (Vector Store)
* **Data Validation**: Pydantic v2 (`pydantic`)
* **User Interfaces**: Streamlit (Web Application) & Terminal CLI

---

## 📁 Project Structure

--Travel Planner/
    |
    |-- data/
    |   |
    |   |-- processed/
    |   |   |-- chroma_db/                    # Auto-generated ChromaDB vector store
    |   |   |-- tourism.db                    # Auto-generated SQLite database
    |   |
    |   |-- raw/
    |   |   |-- dataset_schema.json
    |   |   |-- destination_names.txt
    |   |   |-- india_tourism_dataset.json    # 100-entry Kaggle tourism dataset
    |   |
    |   |-- scripts/
    |   |   |-- ingest_data.py                # SQLite & ChromaDB ingestion pipeline
    |   |
    |-- src/
    |   |
    |   |-- agents/
    |   |   |-- orchestrator_agent.py         # Google ADK agent for JSON extraction
    |   |   |-- presenter_agent.py            # Google ADK agent for itinerary synthesis
    |   |
    |   |-- schemas/
    |   |   |-- query_schema.py               # Pydantic validation schemas
    |   |
    |   |-- tools/
    |   |   |-- search_tools.py               # SQL, Vector, and Hybrid search functions
    |   |
    |   |-- config.py                         # Path constants, model settings, & URLs
    |
    |-- .env                                  #
    |-- main.py                               # Terminal CLI interface
    |-- README.md                             # Project documentation
    |-- requirements.txt                      # Python dependencies

## 🚀 Quickstart Guide

1. Prerequisites
Python 3.10+ installed.

Ollama installed and running locally (http://localhost:11434).

2. Pull Required Models in Ollama
Open your terminal and pull the local LLM and embedding models:

```bash
ollama pull llama2
ollama pull nomic-embed-text
```

3. Virtual Environment & Dependency Installation
Clone the repository, create a virtual environment, and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (macOS/Linux)
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

4. On your CLI run:

```bash
python main.py
```