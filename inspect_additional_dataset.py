import pandas as pd
from pathlib import Path

file_path = Path("dataset/additional/twitter_training.csv")

print("Dataset exists:", file_path.exists())

if file_path.exists():
    print("File size:", file_path.stat().st_size, "bytes")
    df = pd.read_csv(file_path, header=None)
    df.columns = ["tweet_id", "entity", "sentiment", "text"]

    print("\nDataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nSentiment counts:")
    print(df["sentiment"].value_counts(dropna=False))

    print("\nMissing values:")
    print(df.isnull().sum())

else:
    print("Dataset not found. Check the file location.")
