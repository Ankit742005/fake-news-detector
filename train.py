import pandas as pd
import os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

def load_kaggle_data(fake_path, real_path):
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    # Add labels
    fake_df["label"] = "FAKE"
    real_df["label"] = "REAL"

    # Combine
    df = pd.concat([fake_df, real_df], axis=0).reset_index(drop=True)

    # Merge title + text into one feature
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

    # Keep only required columns
    df = df[["text", "label"]]
    return df

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9
        )),
        ("clf", LogisticRegression(max_iter=300))
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", type=str, required=True, help="Path to Fake.csv")
    parser.add_argument("--real", type=str, required=True, help="Path to True.csv")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_kaggle_data(args.fake, args.real)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
