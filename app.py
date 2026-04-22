import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the model pipeline
try:
    model = joblib.load('burnout_prediction_model.joblib')
except FileNotFoundError:
    raise RuntimeError("Model file 'burnout_prediction_model.joblib' not found. Please ensure it's in the correct directory.")

app = FastAPI()

# Configure CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://your-frontend-domain.com",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data schema using Pydantic
class BurnoutPredictRequest(BaseModel):
    day_type: str
    work_hours: float
    screen_time_hours: float
    meetings_count: int
    breaks_taken: int
    after_hours_work: int
    sleep_hours: float
    task_completion_rate: float

# Root endpoint
@app.get("/", summary="Root endpoint")
async def root():
    return {"message": "Welcome to the Work From Home Burnout Risk Prediction API"}

# Prediction endpoint
@app.post("/predict_burnout", summary="Predict burnout risk for an employee")
async def predict_burnout(request_data: BurnoutPredictRequest):
    try:
        # Convert incoming request data to a pandas DataFrame
        input_df = pd.DataFrame([request_data.model_dump()])

        # Make prediction
        prediction = model.predict(input_df)

        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_df).tolist()

        # Get class labels from the trained model's classes_ attribute
        class_labels = model.classes_.tolist() 
        probabilities = dict(zip(class_labels, prediction_proba[0]))

        return {
            "burnout_risk_prediction": prediction[0],
            "probabilities": probabilities
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
