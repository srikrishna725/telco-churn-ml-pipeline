import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_loader import load_data, clean_data
from preprocessor import build_preprocessor
from model import get_random_forest


def main():

    # -----------------------
    # 1. Load and clean data
    # -----------------------
    df = load_data("../data/raw/Telco-Customer-Churn.csv")
    df = clean_data(df)

    # -----------------------
    # 2. Split features & target
    # -----------------------
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -----------------------
    # 3. Build preprocessing
    # -----------------------
    preprocessor = build_preprocessor(X)

    # -----------------------
    # 4. Get model
    # -----------------------
    model = get_random_forest()

    # -----------------------
    # 5. Create full pipeline
    # -----------------------
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # -----------------------
    # 6. Train model
    # -----------------------
    pipeline.fit(X_train, y_train)

    # -----------------------
    # 7. Evaluate model
    # -----------------------
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # -----------------------
    # 8. Save trained model
    # -----------------------
    joblib.dump(pipeline, "../models/final_random_forest_churn_model.pkl")
    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()