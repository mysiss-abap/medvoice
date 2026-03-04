FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiamos requirements primero para cache
COPY requirements.txt .

# Fijamos setuptools a una versión estable que trae pkg_resources
RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir "setuptools==70.3.0"

# Instala deps
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código (incluye pkg_resources.py shim)
COPY . .

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]