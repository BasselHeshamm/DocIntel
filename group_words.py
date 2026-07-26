import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open("sample_invoice.png")
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

words = []
for i in range(len(data["text"])):
    text = data["text"][i]
    conf = int(data["conf"][i])
    if text.strip() == "" or conf == -1:
        continue
    words.append({
        "text": text,
        "left": data["left"][i],
        "top": data["top"][i],
        "width": data["width"][i],
        "height": data["height"][i],
    })

# Sort by top ONLY first. We do NOT sort by left here -- mixing an exact sort key
# (top) with a tolerance-based grouping idea causes bugs, since two words on the
# "same line" rarely share the exact same top pixel value.
words.sort(key=lambda w: w["top"])

# Group into lines using tolerance. Compare each word to the RUNNING AVERAGE top
# of the current line-in-progress (more stable than comparing only to the last word).
LINE_TOLERANCE_PX = 10
lines = []
current_line = [words[0]]

for word in words[1:]:
    avg_top = sum(w["top"] for w in current_line) / len(current_line)
    if abs(word["top"] - avg_top) <= LINE_TOLERANCE_PX:
        current_line.append(word)
    else:
        lines.append(current_line)
        current_line = [word]
lines.append(current_line)

# NOW that words are correctly grouped into lines, sort each line's words
# left-to-right -- this is where "left" should be used, not earlier.
for line in lines:
    line.sort(key=lambda w: w["left"])

GAP_THRESHOLD_PX = 40
print("Detected phrases (grouped words), line by line:\n")
for line in lines:
    phrases = []
    current_phrase = [line[0]]
    for word in line[1:]:
        prev_word = current_phrase[-1]
        gap = word["left"] - (prev_word["left"] + prev_word["width"])
        if gap <= GAP_THRESHOLD_PX:
            current_phrase.append(word)
        else:
            phrases.append(current_phrase)
            current_phrase = [word]
    phrases.append(current_phrase)

    line_output = []
    for phrase in phrases:
        joined_text = " ".join(w["text"] for w in phrase)
        line_output.append(joined_text)
    print("  |  ".join(line_output))