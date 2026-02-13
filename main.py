import pandas as pd
import joblib
import numpy as np
import random
from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------
# 1. LOAD MODELS
# --------------------------------
# Make sure "dish_recommender_model_extended.pkl" is in your project directory
diet_model = joblib.load("lib/models/diet_model.joblib")
food_model = joblib.load("dish_recommender_model_extended.pkl")

app = FastAPI(title="Diet & Food Recommendation API")

# --------------------------------
# 2. REQUEST SCHEMAS
# --------------------------------
class DietRequest(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    goal: str
    diet: str
    disease: str

class FoodRequest(BaseModel):
    diet_category: str  # e.g., "pcos_friendly"
    meal: str           # e.g., "lunch"

# --------------------------------
# 3. DIET PREDICTION
# --------------------------------
@app.post("/predict_diet")
def predict_diet(request: DietRequest):
    height_m = request.height / 100
    bmi = request.weight / (height_m ** 2)

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

# --------------------------------
# 4. FOOD PREDICTION (Randomized Single Pick)
# --------------------------------
@app.post("/predict_food")
def predict_food(request: FoodRequest):
    # 1. Prepare input for the model
    X = pd.DataFrame([{
        "diet_type": request.diet_category,
        "meal_type": request.meal
    }])

    try:
        # 2. Get probabilities for all possible dishes
        probs = food_model.predict_proba(X)[0]
        classes = food_model.named_steps["classifier"].classes_

        # 3. Get indices of the top 10 most likely dishes
        # (This matches your training script's logic)
        top_n = 10
        top_indices = np.argsort(probs)[-top_n:]
        
        # 4. Randomly select exactly ONE dish from the top 10
        random_selection = random.choice(classes[top_indices])

        return {"recommended_food": str(random_selection)}
    
    except Exception as e:
        return {"error": str(e), "message": "Check if diet_category matches training labels."}

# --------------------------------
# RUN COMMAND:
# uvicorn main:app --reload
# --------------------------------
