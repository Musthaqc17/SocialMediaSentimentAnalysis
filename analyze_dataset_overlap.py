from pathlib import Path
import pandas as pd
import re
import hashlib

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"

file1 = DATASET_DIR / "sentiment_dataset.csv"
file2 = DATASET_DIR / "Tweets.csv"


def normalize_text(text):
    """Normalize line endings and repeated whitespace."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset(path):
    df = pd.read_csv(path)

    df["normalized_text"] = df["text"].apply(normalize_text)
    df["normalized_sentiment"] = (
        df["airline_sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


df1 = load_dataset(file1)
df2 = load_dataset(file2)

# Compare text and sentiment together.
pairs1 = set(
    zip(df1["normalized_text"], df1["normalized_sentiment"])
)

pairs2 = set(
    zip(df2["normalized_text"], df2["normalized_sentiment"])
)

common = pairs1 & pairs2
only1 = pairs1 - pairs2
only2 = pairs2 - pairs1

print("=" * 55)
print("NORMALIZED DATASET COMPARISON")
print("=" * 55)

print("Dataset 1 rows:", len(df1))
print("Dataset 2 rows:", len(df2))

print("Unique normalized pairs in Dataset 1:", len(pairs1))
print("Unique normalized pairs in Dataset 2:", len(pairs2))

print("Common pairs:", len(common))
print("Pairs only in Dataset 1:", len(only1))
print("Pairs only in Dataset 2:", len(only2))

print("\nSample pairs only in Dataset 1:")

for text, sentiment in list(only1)[:5]:
    print(f"\nSentiment: {sentiment}")
    print("Text:", text[:250])

print("\nSample pairs only in Dataset 2:")

for text, sentiment in list(only2)[:5]:
    print(f"\nSentiment: {sentiment}")
    print("Text:", text[:250])

print("\nComparison completed.")
