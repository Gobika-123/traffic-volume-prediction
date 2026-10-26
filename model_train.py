import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prepared_traffic_data.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load prepared dataset
print("Loading prepared traffic data...")
df = pd.read_csv(DATA_PATH)

# Select input features and target
features = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day_of_week",
    "month",
    "day_of_month",
]

target = "traffic_volume"

# Keep only required columns and remove invalid rows
df = df.dropna(subset=features + [target])
df = df.sort_values("date_time").reset_index(drop=True)

X = df[features]
y = df[target]

# Chronological split: earlier data for training,
# later data for validation and testing
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

# Train baseline model
print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate on validation data
val_predictions = model.predict(X_val)

val_mae = mean_absolute_error(y_val, val_predictions)
val_rmse = mean_squared_error(y_val, val_predictions) ** 0.5
val_r2 = r2_score(y_val, val_predictions)

print("\nValidation results")
print("-------------------")
print(f"MAE:  {val_mae:.2f}")
print(f"RMSE: {val_rmse:.2f}")
print(f"R2:   {val_r2:.4f}")

# Final evaluation on held-out test data
test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("\nTest results")
print("------------")
print(f"MAE:  {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"R2:   {test_r2:.4f}")

# Save model and feature information
joblib.dump(model, MODEL_DIR / "traffic_volume_model.pkl")
joblib.dump(features, MODEL_DIR / "model_features.pkl")

# Save actual vs predicted results for the test period
results = pd.DataFrame({
    "actual_traffic_volume": y_test.to_numpy(),
    "predicted_traffic_volume": test_predictions
})

results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

print("\nModel training completed successfully!")
print("Saved model to:", MODEL_DIR / "traffic_volume_model.pkl")
print("Saved test predictions to:", OUTPUT_DIR / "test_predictions.csv")
