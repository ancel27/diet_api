from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load models
diet_model = joblib.load("lib/models/diet_model.joblib")
food_model = joblib.load("lib/models/food_model.joblib")

app = FastAPI(title="Diet & Food Recommendation API")

# Request schemas
class DietRequest(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    goal: str       # weight_loss / weight_gain / maintain
    diet: str       # veg / nonveg
    disease: str    # Diabetes, Heart Disease, PCOS, etc.

class FoodRequest(BaseModel):
    diet_category: str  # low_carb_veg / pcos_friendly / heart_healthy, etc.
    diet: str           # veg / nonveg
    meal: str           # breakfast / lunch / dinner

# ---------------- API ENDPOINTS ----------------
@app.post("/predict_diet")
def predict_diet(request: DietRequest):
    X = [[
        request.age,
        request.weight,
        request.height,
        request.gender,
        request.goal,
        request.diet,
        request.disease
    ]]
    pred = diet_model.predict(X)[0]
    return {"diet_category": pred}

@app.post("/predict_food")
def predict_food(request: FoodRequest):
    X = [[
        request.diet_category,
        request.diet,
        request.meal
    ]]
    pred = food_model.predict(X)[0]
    return {"recommended_food": pred}