FROM python:3.11-slim

WORKDIR /app

# dépendances système minimales (xgboost, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code + artefacts (modèles, data pullés par DVC avant build)
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# FastAPI app.app main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
