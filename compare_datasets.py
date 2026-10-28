
from pathlib import Path
import pandas as pd
import hashlib

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"

file1 = DATASET_DIR / "sentiment_dataset.csv"
file2 = DATASET_DIR / "Tweets.csv"

for file_path in [file1, file2]:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    print(f"\n{file_path.name}")
    print("Size:", file_path.stat().st_size, "bytes")
    print("SHA-256:", hashlib.sha256(file_path.read_bytes()).hexdigest())

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print("\nDataset 1 rows:", len(df1))
print("Dataset 2 rows:", len(df2))

print("\nDataset 1 columns:")
print(df1.columns.tolist())

print("\nDataset 2 columns:")
print(df2.columns.tolist())

# Compare complete CSV contents.
identical = df1.equals(df2)

print("\nAre the datasets identical?", identical)

# Compare the text and sentiment columns when available.
required = {"text", "airline_sentiment"}

if required.issubset(df1.columns) and required.issubset(df2.columns):
    texts1 = df1["text"].fillna("").astype(str).tolist()
    texts2 = df2["text"].fillna("").astype(str).tolist()

    labels1 = df1["airline_sentiment"].fillna("").astype(str).tolist()
    labels2 = df2["airline_sentiment"].fillna("").astype(str).tolist()

    print("Tweet text identical and in same order:", texts1 == texts2)
    print("Sentiment labels identical and in same order:", labels1 == labels2)
