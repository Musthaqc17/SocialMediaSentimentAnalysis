
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

for csv_file in DATASET_DIR.glob("*.csv"):
    print(f"\nDataset: {csv_file.name}")

    try:
        df = pd.read_csv(csv_file, nrows=5)

        print("Columns:", df.columns.tolist())
        print("\nFirst five rows:")
        print(df.head())

    except Exception as error:
        print("Could not read file:", error)
