FROM python:3.10-slim

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements-linux.txt .

RUN pip3 install --use-feature=fast-deps --no-cache-dir -r requirements-linux.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
