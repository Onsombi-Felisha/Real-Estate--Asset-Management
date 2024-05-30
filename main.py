from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from typing import List
from fastapi.responses import PlainTextResponse


# Load the trained models
with open('logisticregression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

app = FastAPI()

@app.get("/")
async def read_root():
    message = "Welcome to Prediction of Occupancy Levels and optimising real estate asset management strategies based on key metrics.\n" \
              "Make a POST request to /predict with the following data in the request body: 'Asset Manager', 'Property Type', 'GLA SqFT'.'Gross Value','Occupied SqFt','Vacant SqFt'.\n" \
              "For API documentation, visit: http://localhost:8000/docs"

    return PlainTextResponse(content=message, status_code=200)

# Define the data model for incoming requests
class Item(BaseModel):
    features: List[float]

# Function to translate prediction value to human-readable format
def translate_prediction(prediction: int) -> str:
    if prediction == 1:
        return "Has hundred percent average occupancy ratio"
    else:
        return "Has zero occupancy rate"

# Expose the prediction functionality, make a prediction from the passed data and return the predicted
@app.post("/predict/logistic")
def predict_logistic(item: Item):
    features = np.array(item.features).reshape(1, -1)
    prediction = logistic_model.predict(features)
    translated_prediction = translate_prediction(int(prediction[0]))
    return {"prediction": translated_prediction}
