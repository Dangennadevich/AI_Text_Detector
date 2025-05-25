FROM python:3.11-slim
WORKDIR /app
RUN pip install pandas scikit-learn joblib
COPY train_model.py .
CMD ["python", "train_model.py"]