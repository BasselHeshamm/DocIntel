import re

# re.compile() pre-builds a pattern object -- more efficient if you're going to
# reuse the same pattern many times (e.g., across thousands of documents),
# and it's a common convention you'll see in real codebases.

# Breaking down each pattern piece by piece:

# Dollar amounts: literal '$', then one-or-more digits, optional comma-thousands,
# a literal '.', then exactly two digits.
MONEY_PATTERN = re.compile(r"\$[\d,]+(\.\d{2})?")
# Dates in YYYY-MM-DD format: 4 digits, '-', 2 digits, '-', 2 digits.
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# Invoice numbers: assume a pattern like "INV-2024-0917" --
# letters, then '-', then groups of digits separated by '-'.
INVOICE_NUM_PATTERN = re.compile(r"\bINV-\d{4}-\d{4}\b")


def extract_structured_fields(text):
    """Given raw text, pull out anything matching our rigid, predictable patterns."""
    return {
        "money_amounts": MONEY_PATTERN.findall(text),
        "dates": DATE_PATTERN.findall(text),
        "invoice_numbers": INVOICE_NUM_PATTERN.findall(text),
    }


sample_lines = [
    "TOTAL: $77.00",
    "Invoice #: INV-2024-0917",
    "Date: 2024-11-03",
    "Steel bracket  10  $4.50  $45.00",
]

for line in sample_lines:
    result = extract_structured_fields(line)
    print(f"Input: {line!r}")
    print(f"  -> {result}\n")


