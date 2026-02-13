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
# Get the directory where main.py is located
BASE_DIR = Path(__file__).resolve().parent

# Define absolute paths to your models
# This prevents FileNotFoundError on Render/Cloud hosting
diet_model_path = os.path.join(BASE_DIR, "lib", "models", "diet_model.joblib")
food_model_path = os.path.join(BASE_DIR, "dish_recommender_model_extended.pkl")

# Load the models
try:
    diet_model = joblib.load(diet_model_path)
    food_model = joblib.load(food_model_path)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    # Fallback paths in case they are both in the root folder
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
    # Standard BMI calculation
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
    """
    Takes diet category and meal type, picks one random dish 
    from the top 10 recommended by the Logistic Regression model.
    """
    # 1. Prepare input with correct training column names
    X = pd.DataFrame([{
        "diet_type": request.diet_category,
        "meal_type": request.meal
    }])

    try:
        # 2. Get probabilities for all possible dishes
        probs = food_model.predict_proba(X)[0]
        classes = food_model.named_steps["classifier"].classes_

        # 3. Get indices of the top 10 most likely dishes
        top_n = 10
        # Argsort gives indices of sorted values; we take the last 10
        top_indices = np.argsort(probs)[-top_n:]
        
        # 4. Randomly select exactly ONE dish for the Flutter app
        random_selection = random.choice(classes[top_indices])

        # Clean up the string (replace underscores with spaces for the UI)
        clean_dish_name = str(random_selection).replace("_", " ").title()

        return {"recommended_food": clean_dish_name}
    
    except Exception as e:
        return {"error": str(e), "message": "Ensure input matches trained diet types."}

# --------------------------------
# RUN COMMAND:
# uvicorn main:app --host 0.0.0.0 --port 8000
# --------------------------------
