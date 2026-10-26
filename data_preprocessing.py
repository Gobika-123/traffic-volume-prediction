import pandas as pd
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Automatically find the CSV file
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found. Put your traffic dataset inside the data folder."
    )

if len(csv_files) > 1:
    raise ValueError(
        "More than one CSV file found. Keep only the traffic dataset CSV "
        "inside the data folder."
    )

csv_path = csv_files[0]

print("Loading dataset:", csv_path.name)

# Read the dataset
df = pd.read_csv(csv_path)

print("\nOriginal dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Check required columns
required_columns = ["date_time", "traffic_volume"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}. "
        "Please check your CSV dataset."
    )

# Convert timestamp and traffic volume to valid values
df["date_time"] = pd.to_datetime(
    df["date_time"], errors="coerce"
)

df["traffic_volume"] = pd.to_numeric(
    df["traffic_volume"], errors="coerce"
)

# Report data quality issues
print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove invalid required values
df = df.dropna(subset=["date_time", "traffic_volume"])

# Traffic volume cannot be negative
df = df[df["traffic_volume"] >= 0]

# Remove exact duplicate rows
df = df.drop_duplicates()

# Sort chronologically
df = df.sort_values("date_time").reset_index(drop=True)

# Extract time features
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month
df["day_of_month"] = df["date_time"].dt.day

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save prepared dataset
output_path = OUTPUT_DIR / "prepared_traffic_data.csv"
df.to_csv(output_path, index=False)

print("\nCleaning completed!")
print("Prepared dataset shape:", df.shape)
print("Saved to:", output_path)
print("\nFirst five rows:")
print(df.head())
