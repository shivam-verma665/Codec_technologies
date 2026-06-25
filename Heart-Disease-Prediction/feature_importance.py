import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance")
    print(importance_df)

    plt.figure(figsize=(10,5))

    plt.bar(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xticks(rotation=45)
    plt.title("Heart Disease Feature Importance")
    plt.tight_layout()
    plt.show()