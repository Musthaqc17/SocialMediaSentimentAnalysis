from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent

input_file = project_dir / "dataset" / "additional" / "twitter_training.csv"
output_file = project_dir / "dataset" / "additional" / "twitter_training_clean.csv"

df = pd.read_csv(input_file, header=None)
df.columns = ["tweet_id", "entity", "sentiment", "text"]

print("Original rows:", len(df))

# Remove rows without usable text

df["text"] = df["text"].fillna("").astype(str).str.strip()
df = df[df["text"] != ""]

# Keep only the three supported sentiment classes

df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
df = df[df["sentiment"].isin(["positive", "negative", "neutral"])]

# Save the cleaned dataset

df = df[["text", "sentiment"]].reset_index(drop=True)
df.to_csv(output_file, index=False)

print("\nCleaned rows:", len(df))
print("\nSentiment counts:")
print(df["sentiment"].value_counts())
print("\nSaved to:", output_file)
print("\nDataset preparation completed successfully!")
