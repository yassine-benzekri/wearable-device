from fastapi import FastAPI, UploadFile
import pandas as pd
import pickle

app = FastAPI()

# Load the model
with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)

@app.post("/predict/")
async def predict(file: UploadFile):
    # Load the CSV file
    df = pd.read_csv(file.file)
    
    # Preprocess data (optional)
    # df = preprocess(df)
    
    # Make predictions
    predictions = model.predict(df)
    
    return {"predictions": predictions.tolist()}
