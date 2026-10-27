# استخدام صورة بايثون أساسية خفيفة
FROM python:3.11-slim

# تحديد مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملف متطلبات المشروع
COPY requirements.txt .

# تثبيت المكتبات المطلوبة
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع الأساسية
COPY churn_xgb_model.bin .
COPY predict.py .

# فتح المنفذ 9696
EXPOSE 9696

# تشغيل الخادم عند بدء الحاوية
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]