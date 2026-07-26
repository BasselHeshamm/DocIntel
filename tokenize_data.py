from transformers import AutoTokenizer
from training_data import TRAIN_DATA, LABEL_TO_ID

# AutoTokenizer automatically loads the correct tokenizer for a given model --
# we use the same base model family as our earlier NER model, so the tokenizer
# behaves the same way (same subword-splitting rules) as what we already saw.
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_and_align_labels(words, labels):
    # is_split_into_words=True tells the tokenizer "the input is already a
    # list of individual words, don't try to split raw text yourself."
    tokenized = tokenizer(words, is_split_into_words=True, truncation=True)

    # word_ids() returns, for EACH resulting token, which original word
    # (by index) it came from -- or None for special tokens the model adds
    # automatically (like [CLS] at the start, [SEP] at the end).
    word_ids = tokenized.word_ids()

    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special token (e.g. [CLS], [SEP]) -- not part of our real words at all
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            # First token we've seen for this particular word -- give it the REAL label
            aligned_labels.append(LABEL_TO_ID[labels[word_id]])
        else:
            # We've already labeled a token from this same word -- this is an
            # extra subword piece, so ignore it during training
            aligned_labels.append(-100)
        previous_word_id = word_id

    return tokenized, aligned_labels


# Let's actually SEE this working on our first real training example
words, labels = TRAIN_DATA[0]
print("Original words:", words)
print("Original labels:", labels)

tokenized, aligned_labels = tokenize_and_align_labels(words, labels)

# Convert token IDs back into readable token text, just for us to inspect
tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])

print("\nToken-by-token alignment:")
print(f"{'Token':<12} {'Aligned Label ID':>16}")
for tok, lab in zip(tokens, aligned_labels):
    print(f"{tok:<12} {lab:>16}")