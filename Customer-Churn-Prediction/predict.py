import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("customer_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

print("=" * 60)
print("Customer Churn Prediction")
print("=" * 60)

# Sample customer data
customer = {
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 0,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 70.50,
    "TotalCharges": 850.75
}

# Convert to DataFrame
customer_df = pd.DataFrame([customer])

# Scale features
customer_scaled = scaler.transform(customer_df)

# Predict
prediction = model.predict(customer_scaled)[0]
probability = model.predict_proba(customer_scaled)[0][1]

print(f"\nProbability of Churn: {probability:.2%}")

if prediction == 1:
    print("Prediction: Customer is likely to churn.")
else:
    print("Prediction: Customer is likely to stay.")