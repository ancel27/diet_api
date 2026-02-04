from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load models
diet_model = joblib.load("lib/models/diet_model.joblib")
food_model = joblib.load("lib/models/food_model.joblib")

app = FastAPI(title="Diet & Food Recommendation API")

# ---------------- REQUEST SCHEMAS ----------------
class DietRequest(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    goal: str
    diet: str
    disease: str

class FoodRequest(BaseModel):
    diet_category: str
    diet: str
    meal: str

# ---------------- API ENDPOINTS ----------------
@app.post("/predict_diet")
def predict_diet(request: DietRequest):

    # ✅ Compute BMI (same logic as training)
    height_m = request.height / 100
    bmi = request.weight / (height_m ** 2)

    # ✅ DataFrame with EXACT training columns
    X = pd.DataFrame([{
        "age": request.age,
        "weight": request.weight,
        "height": request.height,
        "bmi": bmi,
        "gender": request.gender,
        "goal": request.goal,
        "diet": request.diet,
        "disease": request.disease
    }])

    pred = diet_model.predict(X)[0]
    return {"diet_category": pred}


@app.post("/predict_food")
def predict_food(request: FoodRequest):
    X = pd.DataFrame([{
        "diet_category": request.diet_category,
        "diet": request.diet,
        "meal": request.meal
    }])

    pred = food_model.predict(X)[0]
    return {"recommended_food": pred}
