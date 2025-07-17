from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/model.pkl")

class Iris(BaseModel):
    features: list

@app.post("/predict")
def predict(iris: Iris):
    prediction = model.predict([iris.features])
    return {"prediction": int(prediction[0])}
