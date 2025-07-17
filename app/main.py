from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from app.model_loader import load_model
import numpy as np

app = FastAPI()
model = load_model()

class Iris(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "Use /docs to test the API."}

@app.post("/predict")
def predict(data: Iris):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}
