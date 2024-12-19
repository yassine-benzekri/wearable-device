import joblib
import pandas as pd

loaded_model = joblib.load("model/wearable_device.pkl")
print("Model loaded successfully.")
# Load sample data
data = pd.read_csv("data/wearable.csv")


# Run inference
predictions = model.predict(data)
print("Predictions:", predictions)
