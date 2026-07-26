import re
import sqlite3
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from google import genai
from pydantic import BaseModel
from typing import Optional

ft_tokenizer = AutoTokenizer.from_pretrained("./finetuned_invoice_ner")
ft_model = AutoModelForTokenClassification.from_pretrained("./finetuned_invoice_ner")
ft_model.eval()

gemini_client = genai.Client()

MONEY_PATTERN = re.compile(r"\$[\d,]+(?:\.\d{2})?")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
INVOICE_NUM_PATTERN = re.compile(r"\bINV-\d{4}-\d{4}\b")


class FallbackFields(BaseModel):
    vendor_name: Optional[str] = None
    total_amount: Optional[str] = None
    invoice_date: Optional[str] = None
    invoice_number: Optional[str] = None


def finetuned_vendor_lookup(line):
    words = line.split()
    if not words:
        return None

    tokenized = ft_tokenizer(words, is_split_into_words=True, truncation=True, return_tensors="pt")
    word_ids = tokenized.word_ids()

    with torch.no_grad():
        outputs = ft_model(**tokenized)

    predicted_ids = outputs.logits.argmax(dim=-1)[0]

    seen = set()
    vendor_words = []
    for word_id, pred_id in zip(word_ids, predicted_ids):
        if word_id is not None and word_id not in seen:
            label = ft_model.config.id2label[pred_id.item()]
            if label in ("B-VENDOR", "I-VENDOR"):
                vendor_words.append(words[word_id])
            seen.add(word_id)

    return " ".join(vendor_words) if vendor_words else None


def extract_fields(raw_text):
    result = {}
    source = {}

    money_matches = MONEY_PATTERN.findall(raw_text)
    if money_matches:
        result["total_amount"] = money_matches[-1]
        source["total_amount"] = "regex"

    date_matches = DATE_PATTERN.findall(raw_text)
    if date_matches:
        result["invoice_date"] = date_matches[0]
        source["invoice_date"] = "regex"

    invoice_num_matches = INVOICE_NUM_PATTERN.findall(raw_text)
    if invoice_num_matches:
        result["invoice_number"] = invoice_num_matches[0]
        source["invoice_number"] = "regex"

    for line in raw_text.strip().split("\n"):
        vendor = finetuned_vendor_lookup(line)
        if vendor:
            result["vendor_name"] = vendor
            source["vendor_name"] = "finetuned"
            break

    needed_fields = ["vendor_name", "total_amount", "invoice_date", "invoice_number"]
    missing = [f for f in needed_fields if f not in result]

    if missing:
        prompt = (
            f"Extract ONLY these fields from the invoice text: {', '.join(missing)}. "
            f"If a field genuinely isn't present in the text, return null for it.\n\n"
            f"Text:\n{raw_text}"
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": FallbackFields,
            },
        )
        llm_result = response.parsed
        for field in missing:
            value = getattr(llm_result, field)
            if value:
                result[field] = value
                source[field] = "llm"

    return result, source


def append_correction_for_retraining(vendor_name):
    if not vendor_name:
        return
    words = vendor_name.split()
    labels = ["B-VENDOR"] + ["I-VENDOR"] * (len(words) - 1)
    entry = {"words": words, "labels": labels}
    with open("human_corrections.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_to_db(raw_text, result, source):
    conn = sqlite3.connect("docintel.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO extractions (
            raw_text, vendor_name, vendor_source,
            total_amount, total_amount_source,
            invoice_date, invoice_date_source,
            invoice_number, invoice_number_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        raw_text,
        result.get("vendor_name"), source.get("vendor_name"),
        result.get("total_amount"), source.get("total_amount"),
        result.get("invoice_date"), source.get("invoice_date"),
        result.get("invoice_number"), source.get("invoice_number"),
    ))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    clean_invoice_text = """
    Acme Corp
    Invoice #: INV-2024-0917
    Date: 2024-11-03
    TOTAL: $77.00
    """

    messy_invoice_text = """
    Remit To: Global Textiles GmbH
    Rechnung Nr: GT-88213
    Faelligkeitsdatum: 2024-09-30
    Gesamtbetrag faellig: EUR 2,450.00
    """

    for label, text in [("Clean invoice", clean_invoice_text), ("Messy/foreign invoice", messy_invoice_text)]:
        print(f"\n=== {label} ===")
        fields, sources = extract_fields(text)
        for k, v in fields.items():
            print(f"  {k}: {v}  (source: {sources[k]})")
        save_to_db(text, fields, sources)
        print("  -> saved to database")