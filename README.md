# DocIntel

A hybrid classical ML + LLM pipeline for extracting structured data (vendor, amount, date, invoice number) from unstructured invoice text, with confidence-based routing and a human-in-the-loop active learning loop.

Built as a learning project to understand production ML system design end-to-end — not just training a model, but the full stack around it: OCR, extraction routing, persistence, an API, async processing, and a feedback loop that improves the model from real corrections.

---

## What it does

1. **OCR** : reads raw text out of an invoice image (Tesseract)
2. **Extraction, cheapest tool first:**
   - Regex for rigid, predictable fields (dollar amounts, dates, invoice numbers)
   - A self-fine-tuned BERT token-classification model for free-text fields (vendor names)
   - An LLM (Gemini, schema-constrained JSON output) as a fallback — only called for fields the cheap tools couldn't confidently extract
3. **Every field is tagged with its source** (`regex` / `finetuned` / `llm` / `human`) — a full audit trail of what produced each value
4. **Async processing**: documents are queued and processed in the background; the API responds immediately with a job ID rather than blocking
5. **Human review loop**: documents with missing or LLM-sourced fields are automatically flagged; a reviewer can correct them through the dashboard, and every correction is saved back into a growing training set
6. **Active learning**: the fine-tuned model can be retrained on the original dataset plus every accumulated human correction, closing the loop from "model gets something wrong" → "human fixes it" → "model gets better"

---

## Architecture

```
Document (image/text)
        │
        ▼
   OCR (Tesseract)
        │
        ▼
┌───────────────────────────────┐
│   Extraction — cheapest first │
│                                │
│  Regex ──► found? ─────► done │
│    │ no                       │
│    ▼                          │
│  Fine-tuned NER ─► found? ──► done │
│    │ no / low confidence      │
│    ▼                          │
│  LLM fallback (Gemini,        │
│  schema-constrained JSON)     │
└───────────────────────────────┘
        │
        ▼
   SQLite (with per-field source tracking)
        │
        ├──► Complete, high-confidence → auto-approved
        │
        └──► Missing / LLM-sourced → Review Queue
                     │
                     ▼
              Human correction (dashboard)
                     │
                     ▼
        Saved as new training example
                     │
                     ▼
              Model retraining
```

---

## Tech stack

- **OCR:** Tesseract (via `pytesseract`)
- **NLP/ML:** PyTorch, Hugging Face Transformers (fine-tuned `bert-base-cased` for token classification)
- **LLM:** Google Gemini API, structured output via Pydantic schema
- **Backend:** FastAPI, async background jobs
- **Storage:** SQLite
- **Frontend:** Single-file HTML/JS dashboard (Tailwind), calling the API directly

---

## Running it locally

```bash
# 1. Clone and set up a virtual environment
git clone https://github.com/BasselHeshamm/DocIntel.git
cd DocIntel
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt

# 2. Install Tesseract OCR separately (not a pip package)
# https://github.com/UB-Mannheim/tesseract/wiki

# 3. Set up your environment
copy start.example.bat start.bat
# edit start.bat and add your Gemini API key (https://aistudio.google.com)

# 4. Set up the database
python db_setup.py

# 5. Train the initial model (or bring your own labeled data in training_data.py)
python train_model.py

# 6. Run
.\start.bat
```

Then open `dashboard.html` directly in your browser. It talks to the API at `http://127.0.0.1:8000`.

---

## Design decisions worth calling out

- **Cascade routing, not "LLM for everything."** Regex and the fine-tuned model handle the majority of fields for free; the LLM is only called for fields that are missing or genuinely ambiguous. This is a deliberate cost/latency tradeoff, not a limitation — the routing logic and its reasoning are in `pipeline.py`.
- **Source tracking on every field**, not just the final value. This is what makes the review queue, the active learning loop, and any future cost/accuracy analysis possible — you can't improve what you can't measure.
- **Confidence isn't blindly trusted.** A model that's confidently wrong is more dangerous than one that flags uncertainty, since a confident wrong answer skips human review by design. The routing threshold reflects that.

## Known limitations / what I'd build next

- The fine-tuned model is trained on a small hand-labeled dataset. Real deployment would need a proper labeled corpus (SROIE/CORD/FUNSD or in-house data)
- Job queue is in-process (FastAPI `BackgroundTasks`), not a real distributed queue (Redis/Celery), fine for a single instance, wouldn't scale horizontally as-is
- No vector search / RAG; document similarity lookup was scoped out for time
- A `Dockerfile` is included but not yet run/tested in a container
