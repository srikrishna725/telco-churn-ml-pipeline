from sklearn.ensemble import RandomForestClassifier


def get_random_forest():
    """
    Return the final tuned Random Forest model.

    This model uses the best hyperparameters
    obtained from GridSearchCV in experimentation.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42
    )

    return model