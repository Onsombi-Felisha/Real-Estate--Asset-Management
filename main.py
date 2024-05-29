from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI
@app.get("/")
async def read_root():
    message = "Welcome to Prediction of Occupancy Levels and optimising real estate asset management strategies based on key metrics.\n" \
              "Make a POST request to /predict with the following data in the request body: 'age', 'sex', 'bmi', 'children', 'smoker', 'region'.\n" \
              "For API documentation, visit: http://localhost:8000/docs"

    return {message}

# Load the trained models
with open('logisticregression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

with open('DecisionTreeClassifier_pickle', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('RandomForestClassifier_pickle', 'rb') as file:
    random_forest_model = pickle.load(file)

app = FastAPI()

# Define the data model for incoming requests
class Item(BaseModel):
    features: list

# Expose the prediction functionality, make a prediction from the passed data and
# return the predicted

@app.post("/predict/logistic")
def predict_logistic(item: Item):
    features = np.array(item.features).reshape(1, -1)
    prediction = logistic_model.predict(features)
    return {"prediction": int(prediction[0])}

@app.post("/predict/decision_tree")
def predict_decision_tree(item: Item):
    features = np.array(item.features).reshape(1, -1)
    prediction = decision_tree_model.predict(features)
    return {"prediction": int(prediction[0])}

@app.post("/predict/random_forest")
def predict_random_forest(item: Item):
    features = np.array(item.features).reshape(1, -1)
    prediction = random_forest_model.predict(features)
    return {"prediction": int(prediction[0])}
