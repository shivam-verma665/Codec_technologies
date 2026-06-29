import joblib
from preprocessing import load_and_preprocess_data

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

import matplotlib.pyplot as plt


# Load processed data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Load trained model
model = joblib.load("customer_churn_model.pkl")

# Predictions
y_pred = model.predict(X_test)

print("=" * 60)
print("Classification Report")
print("=" * 60)

print(classification_report(y_test, y_pred))

print("=" * 60)
print("Confusion Matrix")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# Plot ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# Feature Importance (only if supported)
if hasattr(model, "feature_importances_"):
    feature_names = [
        "gender","SeniorCitizen","Partner","Dependents","tenure",
        "PhoneService","MultipleLines","InternetService",
        "OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies",
        "Contract","PaperlessBilling","PaymentMethod",
        "MonthlyCharges","TotalCharges"
    ]

    import pandas as pd

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    importance.plot(kind="bar")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
else:
    print("\nThis model does not provide feature importance.")