from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from typing import List

# Load the trained models
with open('logisticregression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

app = FastAPI()


# Define the data model for incoming requests
class Item(BaseModel):
    features: List[float]

# Function to translate prediction value to human-readable format
def translate_prediction(prediction: int) -> str:
    if prediction == 1:
        return "hundred average occupancy ratio"
    else:
        return "unknown"  # Add more cases as needed

# Expose the prediction functionality, make a prediction from the passed data and return the predicted
@app.post("/predict/logistic")
def predict_logistic(item: Item):
    features = np.array(item.features).reshape(1, -1)
    prediction = logistic_model.predict(features)
    translated_prediction = translate_prediction(int(prediction[0]))
    return {"prediction": translated_prediction}
