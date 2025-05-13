# LocalTabula: Natural Language Tabular Data Assistant

 
[![License: GPL-3.0](https://img.shields.io/badge/license-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)



**LocalTabula** is a Streamlit app that transforms your spreadsheets into an interactive chat—right on your machine. Drop in an Excel file or a published Google Sheet CSV, then ask questions in plain English. No cloud services, no data leaks: everything runs locally, powered by SQLite and an in‑memory Qdrant vector store.

---

**Why Go Local in an Era of Super-Smart API LLMs?**

Sure, API-driven powerhouses like ChatGPT or Claude can nail SQL generation, and many cloud tools out-of-the-box boast precision that dwarfs lightweight local models. But if you care about privacy, predictability, customization—and even stretching a small budget—local-first is the only way to guarantee your data stays under your control and your costs stay capped.

1. **Data Sovereignty & Compliance**
   Keep everything on-premises so your sensitive data never leaves your firewall. No matter how ironclad an API’s security may sound, nothing beats zero-network-egress for GDPR, HIPAA, or strict corporate policies.

2. **Cost Predictability**
   Ditch per-token billing and surprise overages. After your one-time investment in modest hardware, inference is essentially free—run queries all day without ever watching a meter.

3. **Latency & Reliability**
   Get consistent responses with no reliance on internet connectivity or external service uptime. LocalTabula even thrives in air-gapped or low-bandwidth environments.

4. **Budget-Friendly Hardware**
   Designed for modest rigs: offload a 4-bit–quantized Gemma3-4B model onto a 4 GB GPU, keep the 1.3 B SQL model on CPU, and voilà—you’ve unlocked powerful local inference without a datacenter.

5. **Tunable Accuracy with Compact Models**
   Yes, a 1.3 B-parameter GGUF model might hit \~90% accuracy on straightforward Spider queries—but plunge into nuanced, conversational questions and accuracy can plummet to \~30%. That’s why LocalTabula’s multi-stage pipeline leans heavily on prompt engineering: editable templates, few-shot examples, retry loops, and error-feedback prompts. With those knobs, you’ll guide even a budget-friendly model to production-grade performance.

6. **Full Customization & Extensibility**
   Control every prompt, swap in the latest open-source weights, tweak retry logic, or layer on RAG/agent workflows. You own the roadmap—no vendor lock-in, no forced upgrades.

---

**In Short:** If your top priorities are privacy, predictability, customization—and wringing real accuracy out of small, budget-friendly models—going local isn’t just an option; it’s the only way to keep your data policies—and your wallet—in check. LocalTabula makes it easy to get up and running, even on humble hardware.



---

## What You Can Do

* **Convert & Index tabular data to SQL database:** Clean column names, build a SQLite database, then select and embed text columns automatically.
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
   An LLM polishes your query into a tightly defined SQL prompt. Great for non-experts—skip it if you prefer writing your own query.

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
| SQL Generation       | `generate_sql_*` / `aux_models`           | Swap NL-SQL models, tweak examples, or retry logic       |
| SQL Execution        | `utils._execute_sql_query`                | Modify DB path, pragmas, or timeouts                   |
| Semantic Search      | `init_qdrant_client` / `aux_models`       | Change embedding model, top-k, or distance metric      |
| Summary              | `generate_final_summary` / `prompts.yaml` | Edit tone, detail level, or skip entirely              |

---
During development, I wrapped the core LLM logic in `LLMWrapper`, letting you switch between a local model or an API-based model (via OpenRouter) by calling `LLMWrapper.mode()` and setting the `OPENROUTER_API_KEY` and `OPENROUTER_MODEL_NAME` environment variables.



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

   ⚠️ **GPU Support Requires Advanced Setup**


    You’ll need to build `llama-cpp-python` with the appropriate CMake flags—a nontrivial process. See the README in the **config** folder for some guidance.


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



## **Work in Progress**
2025.05.13

1. **UI & Settings Panel**
   The core pipeline is stable, but the interface remains basic. A dedicated settings page for tweaking prompts and parameters (e.g., retry counts, max tokens) is coming soon.

2. **SQL Command Editor**
   We’re adding an interactive SQL console so you can write and run custom queries alongside the automated pipeline.

3. **Google Colab Support**
   There’s an experimental `main.ipynb` for Colab, with ngrok tunneling to expose the Streamlit app. Because Colab currently runs CUDA 12.5, GPU setup can be tricky—if you plan to use Colab, we recommend the 8B llama-3.1 model for smooth performance.
