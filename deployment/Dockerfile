FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model
COPY artifacts/ ./artifacts/

# Copy app
COPY deployment/app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
