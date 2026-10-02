FROM python:3.11-slim

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model ./model

COPY index.html .
COPY app.py .

EXPOSE 8000

# Run as non-root
RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]