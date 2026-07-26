import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from training_data import TRAIN_DATA, LABEL_LIST, LABEL_TO_ID, ID_TO_LABEL

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(LABEL_LIST),
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
)


def load_human_corrections(path="human_corrections.jsonl"):
    """Reads every human correction and turns it into the same (words, labels)
    format as our original TRAIN_DATA, so we can combine them."""
    examples = []
    try:
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                examples.append((entry["words"], entry["labels"]))
    except FileNotFoundError:
        pass
    return examples


def tokenize_and_align_labels(words, labels):
    tokenized = tokenizer(words, is_split_into_words=True, truncation=True, return_tensors="pt")
    word_ids = tokenized.word_ids()
    aligned_labels = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            aligned_labels.append(LABEL_TO_ID[labels[word_id]])
        else:
            aligned_labels.append(-100)
        previous_word_id = word_id
    return tokenized, torch.tensor([aligned_labels])


human_examples = load_human_corrections()
full_training_set = TRAIN_DATA + human_examples

print(f"Original examples: {len(TRAIN_DATA)}")
print(f"Human-corrected examples: {len(human_examples)}")
print(f"Total training examples: {len(full_training_set)}\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
EPOCHS = 10

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for words, labels in full_training_set:
        tokenized, aligned_labels = tokenize_and_align_labels(words, labels)
        optimizer.zero_grad()
        outputs = model(**tokenized, labels=aligned_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(full_training_set)
    print(f"Epoch {epoch + 1}/{EPOCHS} - average loss: {avg_loss:.4f}")

model.save_pretrained("./finetuned_invoice_ner")
tokenizer.save_pretrained("./finetuned_invoice_ner")
print("\nRetrained model saved to ./finetuned_invoice_ner (overwritten)")