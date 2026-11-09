
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY stroke_prediction_model.bin .
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 9696

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-9696} --workers 1 --threads 2 --timeout 120 predict:app"]
