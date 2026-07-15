from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging 

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Received prediction request: {flower_data}")
    try :
        prediction = model.predict(flower_data)
        logger.info(f"Prediction result: {int(prediction[0])}")
        return int(prediction[0])
    except Exception as e: 
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/health")
def health_check():
     return {"status": "ok"}
    
    
