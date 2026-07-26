FROM python:3.11-slim

# Install Tesseract OCR at the OS level, same as we did manually earlier
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy just the requirements first -- Docker caches this layer, so if only
# your code changes (not dependencies), rebuilds are much faster
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual project files
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]