from PIL import Image, ImageDraw, ImageFont

# Create a blank white "page" -- this is our raw pixel canvas, just like a scanned document
img = Image.new("RGB", (800, 1000), color="white")
draw = ImageDraw.Draw(img)

# On Windows, a reliable built-in font is Arial, found in C:\Windows\Fonts
# We load it at a few different sizes -- font size/boldness is itself a layout signal
font_large = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", 28)
font_normal = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 18)
font_bold_normal = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", 18)

draw.text((50, 40), "Acme Corp", font=font_large, fill="black")
draw.text((50, 75), "123 Industrial Way, Springfield", font=font_normal, fill="black")

draw.text((550, 40), "INVOICE", font=font_large, fill="black")
draw.text((550, 80), "Invoice #: INV-2024-0917", font=font_normal, fill="black")
draw.text((550, 105), "Date: 2024-11-03", font=font_normal, fill="black")

table_y = 200
headers = ["Item", "Qty", "Unit Price", "Line Total"]
header_x = [50, 400, 500, 630]
for text, x in zip(headers, header_x):
    draw.text((x, table_y), text, font=font_bold_normal, fill="black")

draw.line((50, table_y + 25, 750, table_y + 25), fill="black", width=1)

rows = [
    ("Steel bracket", "10", "$4.50", "$45.00"),
    ("Rubber gasket", "25", "$0.80", "$20.00"),
    ("Hex bolt (M6)", "100", "$0.12", "$12.00"),
]
row_y = table_y + 40
for item, qty, price, total in rows:
    draw.text((50, row_y), item, font=font_normal, fill="black")
    draw.text((400, row_y), qty, font=font_normal, fill="black")
    draw.text((500, row_y), price, font=font_normal, fill="black")
    draw.text((630, row_y), total, font=font_normal, fill="black")
    row_y += 30

draw.line((50, row_y + 10, 750, row_y + 10), fill="black", width=1)
draw.text((500, row_y + 30), "TOTAL:", font=font_bold_normal, fill="black")
draw.text((630, row_y + 30), "$77.00", font=font_bold_normal, fill="black")

img.save("sample_invoice.png")
print("Saved sample_invoice.png")