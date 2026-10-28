
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"

file1 = DATASET_DIR / "sentiment_dataset.csv"
file2 = DATASET_DIR / "Tweets.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Compare tweet text
text1 = df1["text"].fillna("").astype(str).str.strip()
text2 = df2["text"].fillna("").astype(str).str.strip()

# Compare tweet text and sentiment together
data1 = pd.DataFrame({
    "text": text1,
    "sentiment": df1["airline_sentiment"].astype(str).str.strip().str.lower()
})

data2 = pd.DataFrame({
    "text": text2,
    "sentiment": df2["airline_sentiment"].astype(str).str.strip().str.lower()
})

print("DATASET COMPARISON")
print("-" * 40)

print("Dataset 1 rows:", len(data1))
print("Dataset 2 rows:", len(data2))

print("\nUnique tweets in Dataset 1:", text1.nunique())
print("Unique tweets in Dataset 2:", text2.nunique())

print("\nExact text matches:", text1.equals(text2))

# Count records in Dataset 1 that also appear in Dataset 2
records1 = set(zip(data1["text"], data1["sentiment"]))
records2 = set(zip(data2["text"], data2["sentiment"]))

common = records1 & records2

print("Unique text-sentiment pairs in Dataset 1:", len(records1))
print("Unique text-sentiment pairs in Dataset 2:", len(records2))
print("Common text-sentiment pairs:", len(common))

print("\nFirst 5 text differences:")

differences = pd.DataFrame({
    "dataset1": text1,
    "dataset2": text2
})

differences = differences[differences["dataset1"] != differences["dataset2"]]

print(differences.head(5).to_string(index=False))

print("\nComparison completed.")
