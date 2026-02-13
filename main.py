import os
import pandas as pd
import joblib
import numpy as np
import random
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

# --------------------------------
# 1. SETUP PATHS & LOAD MODELS
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Correct paths based on your GitHub folder structure
diet_model_path = os.path.join(BASE_DIR, "lib", "models", "diet_model.joblib")
food_model_path = os.path.join(BASE_DIR, "lib", "models", "dish_recommender_model_extended.pkl")

# Load the models with error handling
try:
    diet_model = joblib.load(diet_model_path)
    food_model = joblib.load(food_model_path)
    print("✅ Models loaded successfully from lib/models/")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    diet_model = None
    food_model = None

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
    diet_category: str
    meal: str

# --------------------------------
# 3. DIET PREDICTION
# --------------------------------
@app.post("/predict_diet")
def predict_diet(request: DietRequest):
    if diet_model is None:
        return {"error": "Diet model not found on server"}
        
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
    if food_model is None:
        return {"error": "Food model not found on server"}

    # Prepare input for the model
    X = pd.DataFrame([{
        "diet_type": request.diet_category,
        "meal_type": request.meal
    }])

    try:
        # Get probabilities for all possible dishes
        probs = food_model.predict_proba(X)[0]
        classes = food_model.named_steps["classifier"].classes_

        # Get indices of the top 10 most likely dishes
        top_n = 10
        top_indices = np.argsort(probs)[-top_n:]
        
        # Randomly select exactly ONE dish from the top 10
        random_selection = random.choice(classes[top_indices])
        
        # Clean the name (e.g., "veg_soup" -> "Veg Soup")
        clean_name = str(random_selection).replace("_", " ").title()

        return {"recommended_food": clean_name}
    
    except Exception as e:
        return {"error": str(e), "message": "Check if diet_category matches training labels."}
