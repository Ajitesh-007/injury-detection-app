FROM python:3.11-slim-bookworm

WORKDIR /app

# System libs required by OpenCV & MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer)
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ /app/

# Copy built frontend (must run `npm run build` first in frontend/)
COPY frontend/dist/ /app/../frontend/dist/

# models_data dir — runtime download fills this in
RUN mkdir -p /app/models_data

EXPOSE 8000

# Railway sets PORT env var; fall back to 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
