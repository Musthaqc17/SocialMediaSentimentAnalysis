
from pathlib import Path
import kagglehub
import shutil

# Main project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Destination for the downloaded dataset
DATASET_DIR = PROJECT_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Replace this with the actual Kaggle dataset handle.
# Format: username/dataset-slug
DATASET_HANDLE = "USERNAME/DATASET-SLUG"

# Download dataset from Kaggle
download_path = Path(
    kagglehub.dataset_download(DATASET_HANDLE)
)

print("Dataset downloaded to:", download_path)

# Copy downloaded files into your project dataset folder
for file_path in download_path.rglob("*"):
    if file_path.is_file():
        destination = DATASET_DIR / file_path.name

        if not destination.exists():
            shutil.copy2(file_path, destination)
            print("Copied:", destination.name)

print("Dataset download and copy completed.")
