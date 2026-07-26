from google import genai
from pydantic import BaseModel

client = genai.Client()

# This class defines the EXACT shape of data we want back.
# Each attribute becomes a required field in the JSON the model must produce.
# The type hints (str) tell pydantic -- and the Gemini API -- what kind of
# value is expected for each field.
class InvoiceFields(BaseModel):
    vendor_name: str
    total_amount: str
    invoice_date: str
    invoice_number: str

# This is realistic messy invoice text -- imagine this came from OCR + our
# grouping code, all merged into one blob, for a vendor format we've never
# seen before (unlike our earlier clean synthetic example).
raw_text = """
Remit To: Global Textiles GmbH
Rechnung Nr: GT-88213
Faelligkeitsdatum: 2024-09-30
Gesamtbetrag faellig: EUR 2,450.00
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"Extract the invoice fields from this text:\n\n{raw_text}",
    config={
        "response_mime_type": "application/json",
        "response_schema": InvoiceFields,
    },
)

# .parsed gives us back an actual InvoiceFields OBJECT (not just a raw JSON
# string) -- pydantic has already validated it matches our schema.
result: InvoiceFields = response.parsed

print("Raw JSON text:", response.text)
print("\nParsed fields:")
print("  Vendor:", result.vendor_name)
print("  Total:", result.total_amount)
print("  Date:", result.invoice_date)
print("  Invoice #:", result.invoice_number)