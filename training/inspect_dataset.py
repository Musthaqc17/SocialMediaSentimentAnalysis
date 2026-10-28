
from pathlib import Path
import pandas as pd

# Locate the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

dataset_files = [
    "sentiment_dataset.csv",
    "airline_sentiment.csv",
    "twitter_sentiment.csv",
]

for filename in dataset_files:
    file_path = DATASET_DIR / filename

    print("\n" + "=" * 60)
    print(f"DATASET: {filename}")
    print("=" * 60)

    if not file_path.exists():
        print("File not found:", file_path)
        continue

    try:
        df = pd.read_csv(file_path)

        print("Number of rows:", len(df))
        print("Columns:", df.columns.tolist())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

    except Exception as error:
        print("Error reading dataset:", error)
