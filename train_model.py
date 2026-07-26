import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from training_data import TRAIN_DATA, VAL_DATA, LABEL_LIST, LABEL_TO_ID, ID_TO_LABEL

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# AutoModelForTokenClassification loads the SAME pretrained BERT weights as
# before, but attaches a fresh, untrained final layer sized to OUR number of
# labels (5, since we have O/B-VENDOR/I-VENDOR/B-AMOUNT/I-AMOUNT).
# id2label/label2id let the model's outputs be readable later (e.g. "B-VENDOR"
# instead of just the number 1).
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(LABEL_LIST),
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
)


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


# The optimizer is the thing that actually APPLIES the weight updates.
# AdamW is a standard, widely-used choice -- lr (learning rate) controls how
# BIG each nudge is. Too high -> unstable/overshoots. Too low -> learns too slowly.
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

EPOCHS = 10  # how many full passes over our tiny training set

print("Starting training...\n")
model.train()  # puts the model in "training mode" (affects some internal behavior)

for epoch in range(EPOCHS):
    total_loss = 0
    for words, labels in TRAIN_DATA:
        tokenized, aligned_labels = tokenize_and_align_labels(words, labels)

        optimizer.zero_grad()  # clear leftover gradients from the previous step
        outputs = model(**tokenized, labels=aligned_labels)  # forward pass + loss, in one call
        loss = outputs.loss

        loss.backward()   # backward pass: compute gradients
        optimizer.step()  # apply the weight updates

        total_loss += loss.item()

    avg_loss = total_loss / len(TRAIN_DATA)
    print(f"Epoch {epoch + 1}/{EPOCHS} - average loss: {avg_loss:.4f}")

print("\nTraining complete. Saving model...")
model.save_pretrained("./finetuned_invoice_ner")
tokenizer.save_pretrained("./finetuned_invoice_ner")
print("Saved to ./finetuned_invoice_ner")