# train_model.py

import joblib

from preprocessing import load_and_preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score
)

# Uncomment this if xgboost is installed
from xgboost import XGBClassifier


# Load preprocessed data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Dictionary to store models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

best_model = None
best_model_name = ""
best_accuracy = 0


print("=" * 60)
print("Training Models")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name


# Save Best Model
joblib.dump(best_model, "customer_churn_model.pkl")

print("\n" + "=" * 60)
print(f"Best Model : {best_model_name}")
print(f"Accuracy   : {best_accuracy:.4f}")
print("Model saved as customer_churn_model.pkl")
print("=" * 60)