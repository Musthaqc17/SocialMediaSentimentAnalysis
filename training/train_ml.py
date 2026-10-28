import pandas as pd
import joblib
from preprocessing.clean_text import clean_text

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing.clean_text import clean_text


# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_DIR / "dataset" / "sentiment_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
# 1. Load and combine datasets
    print("Loading datasets...")

airline_df = pd.read_csv(DATASET_PATH)
airline_df = airline_df[["text", "airline_sentiment"]].copy()
airline_df.columns = ["text", "sentiment"]

twitter_path = PROJECT_DIR / "dataset" / "additional" / "twitter_training_clean.csv"
twitter_df = pd.read_csv(twitter_path)

df = pd.concat([airline_df, twitter_df], ignore_index=True)

df["text"] = df["text"].fillna("").astype(str)
df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
df = df[df["sentiment"].isin(["negative", "neutral", "positive"])]

print(f"Combined dataset rows: {len(df)}")
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("Loading datasets...")

airline_df = pd.read_csv(DATASET_PATH)
airline_df = airline_df[["text", "airline_sentiment"]].copy()
airline_df.columns = ["text", "sentiment"]

twitter_path = PROJECT_DIR / "dataset" / "additional" / "twitter_training_clean.csv"
twitter_df = pd.read_csv(twitter_path)

df = pd.concat([airline_df, twitter_df], ignore_index=True)

df["text"] = df["text"].fillna("").astype(str)
df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

df = df[df["sentiment"].isin(["negative", "neutral", "positive"])]

print(f"Combined dataset rows: {len(df)}")
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


    # 3. Remove missing values
df.dropna(subset=["text", "sentiment"], inplace=True)

    # 4. Clean text
print("Cleaning text...")

df["cleaned_text"] = df["text"].apply(clean_text)

    # Remove empty text
df = df[df["cleaned_text"].str.strip() != ""]

    # 5. Separate features and labels
X = df["cleaned_text"]
y = df["sentiment"]

    # 6. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

    # 7. Define models
models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "SVM": LinearSVC(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    }

results = []

    # 8. Train and evaluate each model
for name, classifier in models.items():

        print("\n" + "=" * 55)
        print(f"Training {name}...")

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True
                )
            ),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        # Predictions
        predictions = pipeline.predict(X_test)

        # Evaluation
        accuracy = accuracy_score(y_test, predictions)

        print(f"\n{name} Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=["negative", "neutral", "positive"],
                zero_division=0
            )
        )

        print("Confusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                predictions,
                labels=["negative", "neutral", "positive"]
            )
        )

        # Save model
        model_filename = (
            name.lower().replace(" ", "_") + ".joblib"
        )

        model_path = MODEL_DIR / model_filename

        joblib.dump(pipeline, model_path)

        print(f"Model saved to: {model_path}")

        results.append({
            "model": name,
            "accuracy": accuracy
        })

    # 9. Compare models
    results_df = pd.DataFrame(results)

results_df.sort_values(
        by="accuracy",
        ascending=False,
        inplace=True
    )

results_path = MODEL_DIR / "model_comparison.csv"

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print(results_df.to_string(index=False))

print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()