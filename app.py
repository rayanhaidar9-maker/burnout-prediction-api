import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------
# Load model
# -----------------------
try:
    model = joblib.load("burnout_prediction_model.joblib")
except FileNotFoundError:
    raise RuntimeError("Model file not found. Ensure 'burnout_prediction_model.joblib' is in the project directory.")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()

# -----------------------
# CORS (safe for Lovable)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Lovable + any frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Input schema
# -----------------------
class BurnoutPredictRequest(BaseModel):
    day_type: str
    work_hours: float
    screen_time_hours: float
    meetings_count: int
    breaks_taken: int
    after_hours_work: int
    sleep_hours: float
    task_completion_rate: float

# -----------------------
# Root endpoint
# -----------------------
@app.get("/")
def root():
    return {"message": "Burnout Prediction API is running 🚀"}

# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
def predict_burnout(request_data: BurnoutPredictRequest):
    try:
        input_df = pd.DataFrame([request_data.model_dump()])

        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        class_labels = model.classes_.tolist()
        probabilities = dict(zip(class_labels, prediction_proba[0]))

        return {
            "prediction": prediction[0],
            "probabilities": probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Render PORT fix (IMPORTANT)
# -----------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
