from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging 


# logging for the terminal and for a file to keep records
logging.basicConfig(  
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")  
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
model = joblib.load("model.pkl") # im loading the model ahead of time at the start to avoid constant repeat loading 

class FlowerInput(BaseModel):
    sepal_length : float 
    sepal_width : float 
    petal_length : float 
    petal_width : float 

@app.post("/predict")
def predict_species(flower: FlowerInput):
    # a 2D array is needed for the neccesary input for scikit-learn
    flower_data = [[flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]] 
    logger.info(f"Received prediction request: {flower_data}")
    try :
        prediction = model.predict(flower_data)
        logger.info(f"Prediction result: {int(prediction[0])}")
        return int(prediction[0])
    except Exception as e: 
                # catching unexpected errors and handling it cleanly
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/health")
def health_check():
    # seperate endpoint just to monitor uptime , very basic json return to confirm health
    return {"status": "ok"}
    
    
