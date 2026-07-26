import pytesseract
from PIL import Image

# Tell pytesseract exactly where the Tesseract program lives on disk,
# since Windows' terminal doesn't know its location automatically (the PATH issue from earlier)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open("sample_invoice.png")

# image_to_data() gives us the FULL structured output: each detected word,
# PLUS its bounding box coordinates, PLUS a confidence score.
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

print(f"{'word':<20} {'x':>5} {'y':>5} {'width':>6} {'height':>6} {'confidence':>10}")
print("-" * 65)

n_boxes = len(data["text"])
for i in range(n_boxes):
    word = data["text"][i]
    conf = int(data["conf"][i])

    # Skip empty/whitespace boxes with conf = -1 (Tesseract's way of saying
    # "no real text detected here, ignore this row")
    if word.strip() == "" or conf == -1:
        continue

    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
    print(f"{word:<20} {x:>5} {y:>5} {w:>6} {h:>6} {conf:>10}")