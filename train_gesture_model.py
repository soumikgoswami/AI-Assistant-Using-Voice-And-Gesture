# train_gesture_model.py
"""
Train ML model on collected gesture dataset (CSV).
Outputs gesture_model.pkl
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

DATASET_FILE = "gesture_dataset.csv"
MODEL_FILE = "gesture_model.pkl"


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.drop("gesture", axis=1)
    y = df["gesture"]
    return X, y


def main():
    X, y = load_dataset(DATASET_FILE)

    # Quick class balance report
    print("Classes:")
    print(y.value_counts())

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pipeline: scaler + classifier
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 12, 20],
        "clf__min_samples_split": [2, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=1)
    print("Training with GridSearchCV... this may take a while")
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)

    # Evaluate on test
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    print("\n✅ Test Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save pipeline
    joblib.dump(best, MODEL_FILE)
    print(f"\n💾 Pipeline saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()

