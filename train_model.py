from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "model.pkl"

REAL_LABELS = {"real", "true", "1", "reliable", "legit"}
FAKE_LABELS = {"fake", "false", "0", "unreliable", "hoax"}


def normalize_label(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in REAL_LABELS:
        return "REAL"
    if normalized in FAKE_LABELS:
        return "FAKE"
    raise ValueError(f"Unsupported label value: {value!r}")


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    data = data[["text", "label"]].dropna()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    data["label"] = data["label"].map(normalize_label)

    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain both REAL and FAKE examples.")

    return data


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    missing_columns = {"text", "label"} - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset must include these columns: {missing}")

    return validate_dataset(data)


def article_text_from_columns(data: pd.DataFrame) -> pd.Series:
    if "text" not in data.columns:
        raise ValueError("Dataset must include a text column.")
    if "title" not in data.columns:
        return data["text"].astype(str)

    title = data["title"].fillna("").astype(str).str.strip()
    text = data["text"].fillna("").astype(str).str.strip()
    return (title + " " + text).str.strip()


def load_labeled_file(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    return pd.DataFrame(
        {
            "text": article_text_from_columns(data),
            "label": label,
        }
    )


def load_fake_true_dataset(fake_path: Path, true_path: Path) -> pd.DataFrame:
    fake_data = load_labeled_file(fake_path, "FAKE")
    true_data = load_labeled_file(true_path, "REAL")
    return validate_dataset(pd.concat([fake_data, true_data], ignore_index=True))


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def train_dataframe(data: pd.DataFrame, model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, object]:
    test_size = 0.25 if len(data) >= 8 else 0.5
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=test_size,
        random_state=42,
        stratify=data["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["FAKE", "REAL"])
    accuracy = accuracy_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "model_path": model_path,
        "class_counts": data["label"].value_counts().sort_index(),
    }


def train_model(dataset_path: Path, model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, object]:
    return train_dataframe(load_dataset(dataset_path), model_path)


def train_fake_true_model(
    fake_path: Path,
    true_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, object]:
    return train_dataframe(load_fake_true_dataset(fake_path, true_path), model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fake news detector.")
    parser.add_argument("--data", type=Path, help="CSV file with text and label columns.")
    parser.add_argument("--fake-data", type=Path, help="CSV file where every row is fake news.")
    parser.add_argument("--true-data", type=Path, help="CSV file where every row is real news.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        type=Path,
        help="Where to save the trained joblib pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    using_single_file = args.data is not None
    using_two_files = args.fake_data is not None or args.true_data is not None

    if using_single_file and using_two_files:
        raise SystemExit("Use either --data or --fake-data with --true-data, not both.")
    if using_two_files and not (args.fake_data and args.true_data):
        raise SystemExit("Both --fake-data and --true-data are required for two-file training.")
    if using_two_files:
        metrics = train_fake_true_model(args.fake_data, args.true_data, args.model)
    elif using_single_file:
        metrics = train_model(args.data, args.model)
    else:
        raise SystemExit("Provide --data or both --fake-data and --true-data.")

    print(f"Saved model to: {metrics['model_path']}")
    print("Class counts:")
    print(metrics["class_counts"])
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Confusion matrix labels: FAKE, REAL")
    print(metrics["confusion_matrix"])
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
