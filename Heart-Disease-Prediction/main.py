import pandas as pd

from data_preprocessing import load_and_preprocess_data
from train_model import train_model, evaluate_model
from feature_importance import plot_feature_importance
from predict import predict_disease

DATA_PATH = "h2.csv"

X_train, X_test, y_train, y_test, scaler = \
    load_and_preprocess_data(DATA_PATH)

model = train_model(X_train, y_train)

evaluate_model(
    model,
    X_test,
    y_test
)

df = pd.read_csv(DATA_PATH)

feature_names = df.drop("Heart_Risk", axis=1).columns

plot_feature_importance(
    model,
    feature_names
)

predict_disease(
    model,
    scaler
)