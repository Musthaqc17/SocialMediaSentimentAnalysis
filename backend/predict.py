import joblib
from pathlib import Path

from preprocessing.clean_text import clean_text

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATHS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "SVM": MODEL_DIR / "svm.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}


def predict_sentiment(text, model_name="SVM"):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter valid text.")

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError("No usable text found.")

    model = joblib.load(model_path)

    prediction = model.predict([cleaned_text])[0]

    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "sentiment": prediction,
        "model": model_name,
    }


if __name__ == "__main__":

    samples = [
        "I absolutely love this service!",
        "The experience was terrible and disappointing.",
        "The flight departed at the scheduled time.",
    ]

    for sample in samples:

        result = predict_sentiment(sample)

        print("\nText:", result["original_text"])
        print("Sentiment:", result["sentiment"])
        print("Model:", result["model"])
