FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# herramientas base
RUN pip install --upgrade pip wheel setuptools

# instalar dependencias
RUN pip install -r requirements.txt

# asegurar pkg_resources
RUN pip install setuptools

COPY . .

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]