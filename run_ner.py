from transformers import pipeline

# The `pipeline()` function is Hugging Face's high-level shortcut -- it bundles
# together: loading a pretrained model, loading its matching tokenizer (the tool
# that splits text into tokens the model understands), and running inference,
# all in one call. Under the hood there's a lot happening; pipeline() hides that
# complexity so we can focus on input/output first.
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# "aggregation_strategy='simple'" tells it to do the B-/I- merging step for us
# automatically -- the same stitching logic we wrote by hand in the toy example.

# Some of our grouped phrases from earlier, as plain text
sample_texts = [
    "Acme Corp",
    "123 Industrial Way, Springfield",
    "Invoice #: INV-2024-0917",
    "Date: 2024-11-03",
    "TOTAL: $77.00",
]

for text in sample_texts:
    print(f"\nInput: {text!r}")
    results = ner(text)
    if not results:
        print("  (no entities detected)")
    for ent in results:
        print(f"  {ent['word']!r} -> {ent['entity_group']} (confidence: {ent['score']:.2f})")