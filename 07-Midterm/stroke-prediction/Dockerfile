FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY stroke_prediction_model.bin .

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "--workers", "2", "predict:app"]
