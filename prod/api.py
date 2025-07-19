from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

# Load the trained XGBoost model from JSON
model = xgb.Booster()
model.load_model("outputs/model.json")

# Define the expected input schema
class UpsellInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str

# Initialize FastAPI
app = FastAPI()

@app.post("/predict")
def predict(data: UpsellInput):
    try:
        # Convert input data to a pandas DataFrame
        df = pd.DataFrame([data.dict()])

        # Cast string fields to categorical
        for colname in df.select_dtypes(include=['object']).columns:
            df[colname] = df[colname].astype('category')

        # Convert to DMatrix (if using Booster)
        dmatrix = xgb.DMatrix(df, enable_categorical=True)

        # Make prediction
        prediction = model.predict(dmatrix)

        return {"upsell_probability": float(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")