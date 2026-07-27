import logging
import os
import re
import sqlite3
import string
import json
import unicodedata
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from google import genai
from pydantic import BaseModel
from typing import Optional


def _load_dotenv(path=".env"):
    """Populate os.environ from a local .env file (KEY=VALUE per line),
    without overwriting variables already set in the real environment.
    Avoids a python-dotenv dependency for something this small."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()

ft_tokenizer = AutoTokenizer.from_pretrained("./finetuned_invoice_ner")
ft_model = AutoModelForTokenClassification.from_pretrained("./finetuned_invoice_ner")
ft_model.eval()

gemini_client = genai.Client()

MONEY_PATTERN = re.compile(r"\$[\d,]+(?:\.\d{2})?")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
INVOICE_NUM_PATTERN = re.compile(r"\bINV-\d{4}-\d{4}\b")

# Debug logging: set DOCINTEL_DEBUG=1 to see, for every extraction, the OCR
# text, regex hits, per-token NER labels/confidence, assembled vendor
# candidates, and why non-winning candidates were rejected. Uses the normal
# `logging` module so it composes with however the caller already configures
# logging -- no custom plumbing needed to turn it on or off.
logger = logging.getLogger("docintel.pipeline")
if os.environ.get("DOCINTEL_DEBUG") == "1":
    logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    logger.setLevel(logging.DEBUG)

# The finetuned NER model was trained on a handful of hand-written sentences
# and never saw invoice boilerplate words, so it sometimes tags them as part
# of a vendor span (see select_vendor). This list is a last-step semantic
# safety net used only to trim the EDGES of an already-decoded span -- it is
# not the primary defense, so it can't turn a real multi-word company name
# into an empty result just because one of its words is also common in
# document headers.
DOCUMENT_KEYWORDS = {
    "invoice", "receipt", "statement", "bill", "billing", "remit",
    "remittance", "quote", "quotation", "estimate", "proforma", "purchase",
    "order", "po", "packing", "slip", "delivery", "note", "credit", "memo",
    "confirmation", "sales", "original", "duplicate", "copy", "page", "no",
    "number", "tax", "total", "subtotal", "amount", "balance", "due",
    "terms", "attn", "attention", "date", "qty", "quantity", "description",
    "reference", "ref",
}

# Below this per-token confidence, a predicted VENDOR tag is downgraded to O
# so a single uncertain token can't seed a bogus candidate span.
MIN_TOKEN_CONFIDENCE = 0.5
# Below this confidence, the best surviving vendor candidate is discarded
# entirely rather than returned -- letting the LLM fallback handle it.
MIN_VENDOR_CONFIDENCE = 0.55


class FallbackFields(BaseModel):
    vendor_name: Optional[str] = None
    total_amount: Optional[str] = None
    invoice_date: Optional[str] = None
    invoice_number: Optional[str] = None


def clean_ocr_text(raw_text):
    """Normalize OCR noise before regex/NER ever see the text: fold unicode
    ligatures/smart quotes to plain ASCII-ish forms, drop stray control
    characters, and collapse repeated horizontal whitespace. Newlines are
    preserved deliberately -- vendor candidate selection segments the
    document by line, so collapsing them would merge unrelated fields."""
    text = unicodedata.normalize("NFKC", raw_text).replace(" ", " ")
    text = "".join(ch for ch in text if ch in "\n\t" or ch.isprintable())
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines)


def _strip_punct(word):
    return word.strip(string.punctuation)


def _tag_line_tokens(line):
    """Run the finetuned NER model on one OCR line, returning each word with
    its predicted label and confidence. Low-confidence VENDOR tags are
    downgraded to O here so a single uncertain token can't seed a candidate."""
    words = line.split()
    if not words:
        return []

    tokenized = ft_tokenizer(words, is_split_into_words=True, truncation=True, return_tensors="pt")
    word_ids = tokenized.word_ids()

    with torch.no_grad():
        outputs = ft_model(**tokenized)

    probs = outputs.logits.softmax(dim=-1)[0]
    predicted_ids = probs.argmax(dim=-1)

    tagged = []
    seen = set()
    for word_id, pred_id, prob_row in zip(word_ids, predicted_ids, probs):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        label = ft_model.config.id2label[pred_id.item()]
        confidence = prob_row[pred_id].item()
        if label in ("B-VENDOR", "I-VENDOR") and confidence < MIN_TOKEN_CONFIDENCE:
            label = "O"
        tagged.append((words[word_id], label, confidence))
    return tagged


def _vendor_spans_from_tags(tagged):
    """Group consecutive VENDOR tags into spans, one span per B-VENDOR.

    This boundary-awareness matters: the finetuned model sometimes emits a
    fresh B-VENDOR for an unrelated word (e.g. a document header) that lands
    right after a real vendor name on the same OCR line -- a naive "collect
    every VENDOR-tagged word on this line" approach (the previous
    implementation) would glue the two together into one bogus entity.
    """
    spans = []
    current = []
    for word, label, confidence in tagged:
        if label == "B-VENDOR":
            if current:
                spans.append(current)
            current = [(word, confidence)]
        elif label == "I-VENDOR":
            current.append((word, confidence))
        else:
            if current:
                spans.append(current)
            current = []
    if current:
        spans.append(current)
    return spans


def _is_field_label_token(word):
    """A word ending in ':' is functioning as a form-field label (TOTAL:,
    Terms:, Vendor:, Attn:) rather than as free-standing text. This is a
    layout signal, not a word list -- it catches field labels the finetuned
    model mistags as VENDOR without needing to enumerate every possible
    label a real-world invoice template might use."""
    return word.endswith(":")


def _trim_non_vendor_edges(span):
    """Peel two kinds of false-positive edge tokens off a candidate span:
    (1) known document-structure words (INVOICE, RECEIPT, TOTAL, ...), and
    (2) colon-terminated field labels (TOTAL:, Terms:, Vendor:). Only the
    edges are touched, and only against these narrow signals, so a
    legitimate multi-word company name is never gutted just because one
    interior word happens to be common."""
    def is_edge_noise(word):
        return _is_field_label_token(word) or _strip_punct(word).lower() in DOCUMENT_KEYWORDS

    span = list(span)
    while span and is_edge_noise(span[0][0]):
        span.pop(0)
    while span and is_edge_noise(span[-1][0]):
        span.pop()
    return span


def _score_candidate(candidate):
    # Confidence dominates the score. A small length bonus favors legitimate
    # multi-word names over stray single tokens, and a small position bonus
    # favors earlier lines (vendor names are almost always near the top of
    # an invoice) -- both are tie-breakers, not the primary signal.
    length_bonus = min(len(candidate["words"]), 3) * 0.01
    position_bonus = -candidate["line"] * 0.005
    return candidate["confidence"] + length_bonus + position_bonus


def select_vendor(raw_text, debug=False):
    """Scan every line of the document for vendor candidates (not just the
    first line with a hit), trim known document-header words from candidate
    edges, then return the highest-scoring surviving candidate.

    Returns (vendor_name, confidence) or (None, None) if nothing survives
    filtering, which lets the caller fall back to the LLM.
    """
    lines = [line for line in raw_text.strip().split("\n") if line.strip()]
    candidates = []
    rejected = []

    for line_index, line in enumerate(lines):
        tagged = _tag_line_tokens(line)
        if debug:
            for word, label, confidence in tagged:
                logger.debug("ner token: line=%d word=%r label=%s confidence=%.3f",
                             line_index, word, label, confidence)

        for span in _vendor_spans_from_tags(tagged):
            raw_text_span = " ".join(w for w, _ in span)
            trimmed = _trim_non_vendor_edges(span)

            if not trimmed:
                rejected.append((line_index, raw_text_span, "resolves to only document keywords/field labels"))
                continue

            words = [_strip_punct(w) for w, _ in trimmed]
            # An organization name is essentially never made ENTIRELY of
            # digit-bearing tokens -- that shape belongs to invoice numbers,
            # PO codes, and reference IDs, which the model occasionally
            # mistags when no real vendor name is present to outrank them.
            if all(any(ch.isdigit() for ch in w) for w in words):
                rejected.append((line_index, raw_text_span, "looks like an identifier/code, not an organization name"))
                continue

            candidates.append({
                "line": line_index,
                "words": words,
                "confidence": sum(c for _, c in trimmed) / len(trimmed),
                "trimmed": len(trimmed) != len(span),
                "raw_text": raw_text_span,
            })

    if not candidates:
        if debug:
            logger.debug("vendor: no candidates survived; rejected=%s", rejected)
        return None, None

    best = max(candidates, key=_score_candidate)
    for candidate in candidates:
        if candidate is not best:
            rejected.append((candidate["line"], candidate["raw_text"], "lower-scoring than chosen candidate"))

    if debug:
        logger.debug("vendor candidates: %s", candidates)
        logger.debug("vendor rejected: %s", rejected)
        logger.debug("vendor selected: %r (confidence=%.3f, trimmed=%s)",
                     " ".join(best["words"]), best["confidence"], best["trimmed"])

    if best["confidence"] < MIN_VENDOR_CONFIDENCE:
        if debug:
            logger.debug("vendor: best candidate below MIN_VENDOR_CONFIDENCE (%.2f), discarding",
                         MIN_VENDOR_CONFIDENCE)
        return None, None

    return " ".join(best["words"]), best["confidence"]


def extract_fields(raw_text, debug=False):
    debug = debug or os.environ.get("DOCINTEL_DEBUG") == "1"
    raw_text = clean_ocr_text(raw_text)
    if debug:
        logger.debug("cleaned OCR text:\n%s", raw_text)

    result = {}
    source = {}

    money_matches = MONEY_PATTERN.findall(raw_text)
    if debug:
        logger.debug("regex total_amount matches: %s", money_matches)
    if money_matches:
        result["total_amount"] = money_matches[-1]
        source["total_amount"] = "regex"

    date_matches = DATE_PATTERN.findall(raw_text)
    if debug:
        logger.debug("regex invoice_date matches: %s", date_matches)
    if date_matches:
        result["invoice_date"] = date_matches[0]
        source["invoice_date"] = "regex"

    invoice_num_matches = INVOICE_NUM_PATTERN.findall(raw_text)
    if debug:
        logger.debug("regex invoice_number matches: %s", invoice_num_matches)
    if invoice_num_matches:
        result["invoice_number"] = invoice_num_matches[0]
        source["invoice_number"] = "regex"

    vendor, vendor_confidence = select_vendor(raw_text, debug=debug)
    if vendor:
        result["vendor_name"] = vendor
        source["vendor_name"] = "finetuned"

    needed_fields = ["vendor_name", "total_amount", "invoice_date", "invoice_number"]
    missing = [f for f in needed_fields if f not in result]
    if debug:
        logger.debug("fields missing after regex+NER, falling back to LLM for: %s", missing)

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
                if debug:
                    logger.debug("llm fallback filled %s=%r", field, value)

    if debug:
        logger.debug("final result: %s", result)
        logger.debug("final source: %s", source)

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

    # Regression case for the "Acme Corp / INVOICE" header-merge bug: OCR
    # commonly puts the company name and the document title on the same
    # visual row, which pytesseract linearizes onto a single text line.
    header_merge_text = "Acme Corp INVOICE\nInvoice #: INV-2024-0917\nDate: 2024-11-03\nTOTAL: $77.00"

    # Regression case for "vendor predicted as INVOICE": the document title
    # appears alone, on its own line, above the real vendor name.
    title_before_vendor_text = "INVOICE\nGlobex Trading Co\nInvoice #: INV-2024-0917\nDate: 2024-11-03\nTOTAL: $77.00"

    cases = [
        ("Clean invoice", clean_invoice_text),
        ("Messy/foreign invoice", messy_invoice_text),
        ("Header merged with vendor (same OCR line)", header_merge_text),
        ("Document title line precedes vendor", title_before_vendor_text),
    ]

    for label, text in cases:
        print(f"\n=== {label} ===")
        fields, sources = extract_fields(text)
        for k, v in fields.items():
            print(f"  {k}: {v}  (source: {sources[k]})")
        save_to_db(text, fields, sources)
        print("  -> saved to database")