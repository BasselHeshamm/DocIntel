import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from training_data import VAL_DATA, ID_TO_LABEL

tokenizer = AutoTokenizer.from_pretrained("./finetuned_invoice_ner")
model = AutoModelForTokenClassification.from_pretrained("./finetuned_invoice_ner")
model.eval()

true_positives = 0
false_positives = 0
false_negatives = 0

for words, true_labels in VAL_DATA:
    tokenized = tokenizer(words, is_split_into_words=True, truncation=True, return_tensors="pt")
    word_ids = tokenized.word_ids()

    with torch.no_grad():
        outputs = model(**tokenized)
    predicted_ids = outputs.logits.argmax(dim=-1)[0]

    predicted_labels = []
    seen = set()
    for word_id, pred_id in zip(word_ids, predicted_ids):
        if word_id is not None and word_id not in seen:
            predicted_labels.append(ID_TO_LABEL[pred_id.item()])
            seen.add(word_id)

    # Compare token by token. A "positive" is any non-"O" label (an entity).
    for true_label, pred_label in zip(true_labels, predicted_labels):
        true_is_entity = true_label != "O"
        pred_is_entity = pred_label != "O"

        if pred_is_entity and true_label == pred_label:
            true_positives += 1  # correctly identified AND correctly labeled
        elif pred_is_entity and not true_is_entity:
            false_positives += 1  # model said "entity" but it wasn't one
        elif pred_is_entity and true_is_entity and true_label != pred_label:
            false_positives += 1  # model found an entity but got the TYPE wrong
        elif not pred_is_entity and true_is_entity:
            false_negatives += 1  # model MISSED a real entity

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"True positives:  {true_positives}")
print(f"False positives: {false_positives}")
print(f"False negatives: {false_negatives}")
print(f"\nPrecision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 score:  {f1:.2f}")