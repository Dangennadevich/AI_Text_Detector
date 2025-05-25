from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import joblib
import os
import datetime

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

model_dir = "/data/models"
os.makedirs(model_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"{model_dir}/logreg_model_{timestamp}.pkl"

joblib.dump(model, model_path)
print(f"Model saved to {model_path}")