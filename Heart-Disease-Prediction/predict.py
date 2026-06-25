import numpy as np

def predict_disease(model, scaler):

    print("\nEnter Patient Details")

    chest_pain = float(input("Chest Pain (0/1): "))
    shortness = float(input("Shortness of Breath (0/1): "))
    fatigue = float(input("Fatigue (0/1): "))
    palpitations = float(input("Palpitations (0/1): "))
    dizziness = float(input("Dizziness (0/1): "))
    swelling = float(input("Swelling (0/1): "))
    pain = float(input("Pain Arms/Jaw/Back (0/1): "))
    cold_sweats = float(input("Cold Sweats/Nausea (0/1): "))
    high_bp = float(input("High BP (0/1): "))
    high_chol = float(input("High Cholesterol (0/1): "))
    diabetes = float(input("Diabetes (0/1): "))
    smoking = float(input("Smoking (0/1): "))
    obesity = float(input("Obesity (0/1): "))
    sedentary = float(input("Sedentary Lifestyle (0/1): "))
    family = float(input("Family History (0/1): "))
    stress = float(input("Chronic Stress (0/1): "))
    gender = float(input("Gender (0=Female, 1=Male): "))
    age = float(input("Age: "))

    data = np.array([[
        chest_pain,
        shortness,
        fatigue,
        palpitations,
        dizziness,
        swelling,
        pain,
        cold_sweats,
        high_bp,
        high_chol,
        diabetes,
        smoking,
        obesity,
        sedentary,
        family,
        stress,
        gender,
        age
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        print("\nHigh Risk of Heart Disease")
    else:
        print("\nLow Risk of Heart Disease")