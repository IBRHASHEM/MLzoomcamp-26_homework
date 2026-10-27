# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy project requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model and prediction service
COPY churn_xgb_model.bin .
COPY predict.py .

# Expose port 9696
EXPOSE 9696

# Start the web server on container startup
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]