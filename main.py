from fastapi import FastAPI 
from pydantic import BaseModel
import joblib


app = FastAPI()
model = joblib.load("model.pkl")

class FlowerInput(BaseModel):
    sepal_length : float 
    sepal_width : float 
    petal_length : float 
    petal_width : float 

@app.post("/predict")
def predict_species(flower: FlowerInput):
    flower_data = [[flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]] 
    prediction = model.predict(flower_data)
    return int(prediction[0])