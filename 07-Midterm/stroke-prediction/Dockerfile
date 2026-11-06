
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY stroke_prediction_model.bin .

EXPOSE 9696

CMD ["python", "predict.py"]
