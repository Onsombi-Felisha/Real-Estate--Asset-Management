from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained models
with open('logisticregression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

with open('DecisionTreeClassifier_pickle', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('RandomForestClassifier_pickle', 'rb') as file:
    random_forest_model = pickle.load(file)