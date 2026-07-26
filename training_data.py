# A small, hand-labeled dataset. Each example: (list of words, list of labels).
# In a real project this would come from SROIE/CORD/FUNSD or human-labeled invoices --
# we're hand-writing a tiny set ourselves so we control everything and can see
# exactly what's happening at each step.

TRAIN_DATA = [
    (["Acme", "Corp", "billed", "you", "$77"],
     ["B-VENDOR", "I-VENDOR", "O", "O", "B-AMOUNT"]),

    (["Global", "Textiles", "GmbH", "sent", "an", "invoice", "for", "$2450"],
     ["B-VENDOR", "I-VENDOR", "I-VENDOR", "O", "O", "O", "O", "B-AMOUNT"]),

    (["Springfield", "Manufacturing", "charged", "$310"],
     ["B-VENDOR", "I-VENDOR", "O", "B-AMOUNT"]),

    (["Please", "pay", "Bright", "Steel", "Ltd", "$99"],
     ["O", "O", "B-VENDOR", "I-VENDOR", "I-VENDOR", "B-AMOUNT"]),
]

# A SEPARATE validation set -- different vendors/sentences than training,
# so we can honestly check whether the model generalizes.
VAL_DATA = [
    (["Northwind", "Traders", "invoiced", "$150"],
     ["B-VENDOR", "I-VENDOR", "O", "B-AMOUNT"]),

    (["You", "owe", "Fabrikam", "Inc", "$500"],
     ["O", "O", "B-VENDOR", "I-VENDOR", "B-AMOUNT"]),
]

# The full list of possible labels, and a mapping to/from integer IDs --
# models work with numbers internally, not text labels directly.
LABEL_LIST = ["O", "B-VENDOR", "I-VENDOR", "B-AMOUNT", "I-AMOUNT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}