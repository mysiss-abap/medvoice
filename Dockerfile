FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-speechmatics.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install -r requirements-speechmatics.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]