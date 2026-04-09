from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from flask import Flask, jsonify, render_template, request

from utils.helper import MIN_TEXT_CHARS, PredictionError, clean_whitespace, extract_article_text


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = Flask(__name__)
@app.route('/')
def home():
    return "App chal raha hai 🚀"

def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise PredictionError(
            "Model not found. Run `python train_model.py --data data/sample_news.csv` before predicting."
        )
    return joblib.load(MODEL_PATH)


def predict_text(text: str) -> dict[str, Any]:
    text = clean_whitespace(text)
    if not text:
        raise PredictionError("Please paste article text or enter a news article link.")
    if len(text) < MIN_TEXT_CHARS:
        raise PredictionError("Please provide a longer article text for a more reliable prediction.")

    model = load_model()
    label = str(model.predict([text])[0]).upper()

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        class_names = [str(name).upper() for name in model.classes_]
        label_index = class_names.index(label)
        confidence = round(float(probabilities[label_index]) * 100, 2)

    return {
        "label": label.title(),
        "confidence": confidence,
        "disclaimer": "This tool gives an ML-based estimate, not a verified fact-check.",
    }


def prediction_from_payload(payload: dict[str, str]) -> dict[str, Any]:
    text = payload.get("text", "").strip()
    url = payload.get("url", "").strip()

    if text and url:
        raise PredictionError("Please use either article text or a link, not both at once.")
    if url:
        article_text = extract_article_text(url)
        result = predict_text(article_text)
        result["source"] = "link"
        result["extracted_preview"] = article_text[:280]
        return result

    result = predict_text(text)
    result["source"] = "text"
    return result


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/predict")
def predict() -> Any:
    payload = {
        "text": request.form.get("text", ""),
        "url": request.form.get("url", ""),
    }

    try:
        result = prediction_from_payload(payload)
    except PredictionError as exc:
        return render_template("index.html", error=str(exc), form=payload), 400

    return render_template("index.html", result=result, form=payload)


@app.post("/api/predict")
def api_predict() -> Any:
    payload = request.get_json(silent=True) or {}
    payload = {"text": str(payload.get("text", "")), "url": str(payload.get("url", ""))}

    try:
        return jsonify(prediction_from_payload(payload))
    except PredictionError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True)
