# LocalTabula: Natural Language Tabular Data Assistant

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# LocalTabula: Your Local-First Tabular Data Assistant

**LocalTabula** is a Streamlit app that transforms your spreadsheets into an interactive chat—right on your machine. Drop in an Excel file or a published Google Sheet CSV, then ask questions in plain English. No cloud services, no data leaks: everything runs locally, powered by SQLite and an in‑memory Qdrant vector store.

---

## Why Local First?

Many “chat-to-data” tools live in the cloud and hide their inner workings behind rigid UIs. LocalTabula flips the script:

* **Full Ownership:** Your data never leaves your computer.
* **Customizable Pipeline:** Tweak prompts, swap models, and tune indexing to fit your needs.
* **Multi-Stage Architecture:** Each step validates and refines, so you get accurate, explainable results.

---

## What You Can Do

* **Upload & Index:** Clean column names, build a SQLite database, then select and embed text columns automatically.
* **Natural-Language Queries:** Ask things like “What’s our Q1 revenue by region?” or “Find products like X.”
* **Smart Routing:** The app decides—SQL or semantic search—then packages schema and sample rows for context.
* **Inspect & Tweak:** Expand SQL statements, preview raw rows or embeddings, and re-index on demand.
* **Offline or API:** During development, point to OpenRouter; in production, run purely local GGUF models (with optional GPU acceleration).

---

## Under the Hood: The 5-Stage Query Pipeline

1. **Preflight & Normalization**
   Convert any user question into clear English so local models (or your custom ones) perform at their best. Swap or disable this step via `prompts.yaml`.

2. **Route & Schema Prep**
   Choose SQL or semantic mode based on your question’s structure. Attach the relevant table’s schema and a few sample rows to guide the LLM.

3. **Prompt Refinement (Optional)**
   An LLM polishes your query into a tightly defined SQL prompt. Great for non-experts—skip it if you prefer writing your own SQL.

4. **Execution & Fallback**

   * **SQL Mode:** Generate, validate, and run SQL against SQLite. On error, loop once for correction.
   * **Semantic Mode:** Embed your question, run a vector search over Qdrant, and return top text snippets.
     If your primary route returns nothing, the other route kicks in as a safety net.

5. **Natural-Language Summary**
   Feed raw results back into an LLM to produce a conversational answer—no more staring at raw tables. Edit the final summary prompt in `prompts.yaml` or disable it to see raw outputs.

---

## Tweak These Knobs

All configurations live in **`.env`**, **`config/prompts.yaml`**, and the helper functions in **`utils.py`**:

| Stage                | File / Function                           | Customize                                              |
| -------------------- | ----------------------------------------- | ------------------------------------------------------ |
| Language Normalizer  | `prompts.yaml`                            | Swap translation logic or disable                      |
| DB Selection         | `select_database_id` / `prompts.yaml`     | Change sample size, prompt template, or fallback logic |
| Refinement & Routing | `refine_and_select` / `prompts.yaml`      | Adjust few-shots, temperature, or force-mode flags     |
| SQL Generation       | `generate_sql_*` / `aux_models`           | Swap GGUF models, tweak examples, or retry logic       |
| SQL Execution        | `utils._execute_sql_query`                | Modify DB path, pragmas, or timeouts                   |
| Semantic Search      | `init_qdrant_client` / `aux_models`       | Change embedding model, top-k, or distance metric      |
| Summary              | `generate_final_summary` / `prompts.yaml` | Edit tone, detail level, or skip entirely              |

---

## Getting Started

1. **Clone & Activate**

   ```bash
   git clone <repo-url> && cd <repo>
   python -m venv venv && source venv/bin/activate
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   For GPU support, reinstall `llama-cpp-python` with proper CMake flags.

3. **Configure**
   Copy `.env.example` to `.env` and set:

   * `DEVELOPMENT_MODE` (true for OpenRouter, false for local GGUF)
   * Paths or repo IDs for your GGUF models
   * API keys (if using OpenRouter)

4. **Run**

   ```bash
   streamlit run app.py
   ```

   Visit `http://localhost:8501` and start chatting with your data!

---

For GPU support, read the README file under config folder

**Work in Progress:** The core pipeline is stable, but the UI and settings panel are evolving. Contributions and feedback are welcome—this is your homegrown tool!
