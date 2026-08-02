from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sqlite3
import uuid
import pytesseract
from PIL import Image
import io

from pipeline import (
    extract_fields,
    save_to_db,
    append_correction_for_retraining,
    assess_document_validity,
    ensure_schema,
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def migrate_db_schema():
    conn = sqlite3.connect("docintel.db")
    ensure_schema(conn)
    conn.close()


class ExtractRequest(BaseModel):
    raw_text: str


jobs = {}


def _mean_ocr_confidence(image):
    """Average Tesseract word-level confidence (0-100), ignoring the -1
    placeholder rows Tesseract emits for non-text regions. Returns None if
    nothing was recognized at all, so callers can treat "no signal" as
    distinct from "confidently blank"."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data["conf"] if int(c) != -1]
    return sum(confidences) / len(confidences) if confidences else None


def process_document(job_id: str, raw_text: str, ocr_confidence: Optional[float] = None):
    jobs[job_id]["status"] = "processing"

    is_valid, rejection_reason = assess_document_validity(raw_text, ocr_confidence=ocr_confidence)
    if not is_valid:
        save_to_db(raw_text, {}, {}, is_valid_document=False, rejection_reason=rejection_reason)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "is_valid_document": False,
            "rejection_reason": rejection_reason,
            "fields": {},
            "sources": {},
        }
        return

    fields, sources = extract_fields(raw_text)
    save_to_db(raw_text, fields, sources, is_valid_document=True)
    jobs[job_id]["status"] = "done"
    jobs[job_id]["result"] = {
        "is_valid_document": True,
        "rejection_reason": None,
        "fields": fields,
        "sources": sources,
    }


@app.post("/extract")
def extract_endpoint(request: ExtractRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None}
    background_tasks.add_task(process_document, job_id, request.raw_text)
    return {"job_id": job_id, "status": "queued"}


@app.post("/extract/image")
async def extract_from_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    raw_text = pytesseract.image_to_string(image)
    ocr_confidence = _mean_ocr_confidence(image)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None}
    background_tasks.add_task(process_document, job_id, raw_text, ocr_confidence)

    return {"job_id": job_id, "status": "queued", "ocr_text": raw_text}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        return {"error": "job not found"}
    return jobs[job_id]


@app.get("/documents")
def get_all_documents():
    conn = sqlite3.connect("docintel.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM extractions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/review/pending")
def get_pending_review():
    conn = sqlite3.connect("docintel.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Rejected (is_valid_document = 0) documents are deliberately excluded:
    # there's nothing to "correct" on a document that wasn't an invoice to
    # begin with. They're surfaced separately, as rejection notices, not as
    # rows that look like ordinary incomplete extractions.
    cursor.execute("""
        SELECT * FROM extractions
        WHERE is_valid_document = 1
          AND (vendor_name IS NULL OR vendor_source = 'llm'
           OR total_amount IS NULL OR total_amount_source = 'llm'
           OR invoice_date IS NULL OR invoice_date_source = 'llm'
           OR invoice_number IS NULL OR invoice_number_source = 'llm')
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/documents/rejected")
def get_rejected_documents():
    conn = sqlite3.connect("docintel.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM extractions WHERE is_valid_document = 0 ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


class ReviewCorrection(BaseModel):
    id: int
    vendor_name: Optional[str] = None
    total_amount: Optional[str] = None
    invoice_date: Optional[str] = None
    invoice_number: Optional[str] = None


@app.post("/review/submit")
def submit_review(correction: ReviewCorrection):
    conn = sqlite3.connect("docintel.db")
    cursor = conn.cursor()

    source_column_names = {
        "vendor_name": "vendor_source",
        "total_amount": "total_amount_source",
        "invoice_date": "invoice_date_source",
        "invoice_number": "invoice_number_source",
    }

    updates = []
    values = []
    for field in ["vendor_name", "total_amount", "invoice_date", "invoice_number"]:
        value = getattr(correction, field)
        if value is not None:
            updates.append(f"{field} = ?")
            updates.append(f"{source_column_names[field]} = ?")
            values.append(value)
            values.append("human")

    if updates:
        # A human providing field values is itself evidence the document
        # IS real, even if it was auto-rejected -- so a correction clears
        # any prior rejection instead of leaving it stuck.
        updates.append("is_valid_document = ?")
        updates.append("rejection_reason = ?")
        values.append(1)
        values.append(None)

        values.append(correction.id)
        sql = f"UPDATE extractions SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(sql, values)
        conn.commit()

    conn.close()

    if correction.vendor_name:
        append_correction_for_retraining(correction.vendor_name)

    return {"status": "updated", "id": correction.id}