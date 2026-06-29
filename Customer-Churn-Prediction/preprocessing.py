import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data():

    # Load dataset
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    print("Dataset Loaded Successfully")
    print(df.shape)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Remove Customer ID
    df = df.drop("customerID", axis=1)

    # Encode categorical columns
    encoder = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = encoder.fit_transform(df[col])

    # Features and Target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "scaler.pkl")

    return X_train, X_test, y_train, y_test