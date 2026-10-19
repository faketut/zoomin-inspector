FROM python:3.12-slim

# OpenCV headless wheels still need these system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 7860
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
