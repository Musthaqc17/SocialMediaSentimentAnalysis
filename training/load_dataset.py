import pandas as pd
from pathlib import Path

# Get the main project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset location
DATASET_PATH = PROJECT_DIR / "dataset" / "sentiment_dataset.csv"

# Check whether the dataset exists
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Please download the CSV and place it inside the dataset folder."
    )

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("\n========== DATASET INFORMATION ==========")

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSentiment distribution:")

if "airline_sentiment" in df.columns:
    print(df["airline_sentiment"].value_counts())
else:
    print("Sentiment column not found. Check the column names.")

print("\nDataset loaded successfully!")