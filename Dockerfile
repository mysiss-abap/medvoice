FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-speechmatics.txt .

# ✅ Pip + wheel OK, pero SETUPTOOLS PINNED (incluye pkg_resources)
RUN pip install --upgrade pip wheel && \
    pip install "setuptools==70.3.0"

RUN pip install -r requirements-speechmatics.txt

# ✅ Re-forzar setuptools al final (por si alguna lib lo cambió)
RUN pip install --force-reinstall "setuptools==70.3.0"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]