import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from training_data import VAL_DATA, ID_TO_LABEL

tokenizer = AutoTokenizer.from_pretrained("./finetuned_invoice_ner")
model = AutoModelForTokenClassification.from_pretrained("./finetuned_invoice_ner")
model.eval()

for words, true_labels in VAL_DATA:
    tokenized = tokenizer(words, is_split_into_words=True, truncation=True, return_tensors="pt")
    word_ids = tokenized.word_ids()

    with torch.no_grad():
        outputs = model(**tokenized)

    predicted_ids = outputs.logits.argmax(dim=-1)[0]

    print(f"\nWords: {words}")
    print(f"True labels:      {true_labels}")

    predicted_labels = []
    seen_word_ids = set()
    for word_id, pred_id in zip(word_ids, predicted_ids):
        if word_id is not None and word_id not in seen_word_ids:
            predicted_labels.append(ID_TO_LABEL[pred_id.item()])
            seen_word_ids.add(word_id)

    print(f"Predicted labels: {predicted_labels}")