# Synthetic labeled dataset for the vendor/amount NER model.
#
# The original version of this file had 4 hand-written examples. That was
# nowhere near enough for the model to learn what ISN'T a vendor -- it had
# never seen a document keyword (INVOICE, TOTAL, ...) or a customer name
# next to a vendor span, so it generalized by "any capitalized phrase near
# a vendor context is probably also part of the vendor," which is exactly
# what caused real extraction bugs (vendor = "Acme Corp INVOICE", vendor =
# "INVOICE"). This version crosses a varied pool of vendor names against a
# set of sentence templates -- several of which deliberately place a
# document keyword immediately before/after the vendor span with an O
# label -- plus a block of pure-negative examples with no vendor at all.

LABEL_LIST = ["O", "B-VENDOR", "I-VENDOR", "B-AMOUNT", "I-AMOUNT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def _vendor_labels(vendor_words):
    return ["B-VENDOR"] + ["I-VENDOR"] * (len(vendor_words) - 1)


# TRAIN and VAL use completely disjoint companies so validation genuinely
# measures generalization to unseen vendors, not memorization.
#
# Deliberately varied: single-word ("Nike"), multi-word ("Acme Corp"),
# numeric ("3M", "84 Lumber", "7-Eleven"), and Inc/LLC/GmbH/Ltd/Corp/AG/PLC
# suffixes.
TRAIN_VENDORS = [
    ["Acme", "Corp"],
    ["Global", "Textiles", "GmbH"],
    ["Springfield", "Manufacturing"],
    ["Bright", "Steel", "Ltd"],
    ["Contoso", "Solutions"],
    ["Initech", "Systems"],
    ["Wayne", "Enterprises"],
    ["Cyberdyne", "Systems", "Corp"],
    ["Nike"],
    ["3M"],
    ["84", "Lumber"],
    ["Massive", "Dynamic", "AG"],
    ["Umbrella", "Corporation"],
    ["Prestige", "Worldwide"],
    ["Oceanic", "Airlines"],
    ["Soylent", "Corp"],
    ["Gringotts", "Bank"],
    ["Blue", "Sun", "Corporation"],
]

VAL_VENDORS = [
    ["Northwind", "Traders", "LLC"],
    ["Fabrikam", "Inc"],
    ["Hooli", "Technologies"],
    ["7-Eleven"],
    ["Stark", "Industries"],
    ["Wonka", "Chocolate", "Factory"],
    ["Zephyr", "Airlines", "PLC"],
]


# Each template takes a vendor's word list and returns (words, labels).
#
# The `*_then_*` and `*_before_*` templates are the key fix: they teach the
# model that INVOICE/TOTAL/RECEIPT/STATEMENT are O even when they sit
# immediately next to a real vendor span -- exactly the adjacency the old
# model got wrong (see pipeline.py's vendor selection for the runtime-side
# mitigation this data-level fix complements).

def _bare(vendor):
    return vendor, _vendor_labels(vendor)


def _billed_you(vendor):
    words = vendor + ["billed", "you", "$77"]
    return words, _vendor_labels(vendor) + ["O", "O", "B-AMOUNT"]


def _sent_invoice_for(vendor):
    words = vendor + ["sent", "an", "invoice", "for", "$2450"]
    return words, _vendor_labels(vendor) + ["O", "O", "O", "O", "B-AMOUNT"]


def _charged(vendor):
    words = vendor + ["charged", "$310"]
    return words, _vendor_labels(vendor) + ["O", "B-AMOUNT"]


def _please_pay(vendor):
    words = ["Please", "pay"] + vendor + ["$99"]
    return words, ["O", "O"] + _vendor_labels(vendor) + ["B-AMOUNT"]


def _invoiced(vendor):
    words = vendor + ["invoiced", "$150"]
    return words, _vendor_labels(vendor) + ["O", "B-AMOUNT"]


def _you_owe(vendor):
    words = ["You", "owe"] + vendor + ["$500"]
    return words, ["O", "O"] + _vendor_labels(vendor) + ["B-AMOUNT"]


def _remit_to(vendor):
    words = ["Remit", "To:"] + vendor
    return words, ["O", "O"] + _vendor_labels(vendor)


def _bill_from(vendor):
    words = ["Bill", "From:"] + vendor
    return words, ["O", "O"] + _vendor_labels(vendor)


def _pay_to(vendor):
    words = ["Pay", "To:"] + vendor
    return words, ["O", "O"] + _vendor_labels(vendor)


def _sold_by(vendor):
    words = ["Sold", "By:"] + vendor
    return words, ["O", "O"] + _vendor_labels(vendor)


def _vendor_label_prefix(vendor):
    words = ["Vendor:"] + vendor
    return words, ["O"] + _vendor_labels(vendor)


def _vendor_then_invoice(vendor):
    words = vendor + ["INVOICE"]
    return words, _vendor_labels(vendor) + ["O"]


def _vendor_then_total(vendor):
    words = vendor + ["TOTAL"]
    return words, _vendor_labels(vendor) + ["O"]


def _vendor_then_receipt(vendor):
    words = vendor + ["RECEIPT"]
    return words, _vendor_labels(vendor) + ["O"]


def _invoice_then_vendor(vendor):
    words = ["INVOICE"] + vendor
    return words, ["O"] + _vendor_labels(vendor)


def _statement_then_vendor(vendor):
    words = ["STATEMENT"] + vendor
    return words, ["O"] + _vendor_labels(vendor)


TEMPLATES = [
    _bare, _billed_you, _sent_invoice_for, _charged, _please_pay,
    _invoiced, _you_owe, _remit_to, _bill_from, _pay_to, _sold_by,
    _vendor_label_prefix, _vendor_then_invoice, _vendor_then_total,
    _vendor_then_receipt, _invoice_then_vendor, _statement_then_vendor,
]


def _build(vendors, templates, offset=0):
    """Apply two different templates per vendor (spread apart in the list)
    so each vendor shows up in more than one sentence shape without a full
    cross product blowing up the dataset size."""
    examples = []
    n = len(templates)
    for i, vendor in enumerate(vendors):
        examples.append(templates[(i + offset) % n](vendor))
        examples.append(templates[(i + offset + n // 2) % n](vendor))
    return examples


TRAIN_DATA = _build(TRAIN_VENDORS, TEMPLATES)
VAL_DATA = _build(VAL_VENDORS, TEMPLATES, offset=3)

# Negative examples: no vendor present at all. These matter as much as the
# positive examples above -- without them the model has never seen a
# capitalized document keyword or a customer name on its own and simply
# assumes any proper-noun-shaped phrase must be the vendor.
NEGATIVE_EXAMPLES = [
    (["INVOICE"], ["O"]),
    (["RECEIPT"], ["O"]),
    (["STATEMENT"], ["O"]),
    (["TOTAL", "DUE"], ["O", "O"]),
    (["Sub-Total"], ["O"]),
    (["Bill", "To:", "John", "Smith"], ["O", "O", "O", "O"]),
    (["Ship", "To:", "Jane", "Doe"], ["O", "O", "O", "O"]),
    (["Attn:", "Accounts", "Payable"], ["O", "O", "O"]),
    (["Thank", "you", "for", "your", "business"], ["O", "O", "O", "O", "O"]),
    (["Payment", "is", "due", "within", "30", "days"], ["O", "O", "O", "O", "O", "O"]),
    (["Description", "Qty", "Unit", "Price"], ["O", "O", "O", "O"]),
    (["Date:", "2024-11-03"], ["O", "O"]),
    (["Invoice", "#:", "INV-2024-0917"], ["O", "O", "O"]),
    (["Terms:", "Net", "30"], ["O", "O", "O"]),
]

TRAIN_DATA += NEGATIVE_EXAMPLES[:10]
VAL_DATA += NEGATIVE_EXAMPLES[10:]
