import os
import pandas as pd
import joblib
import numpy as np
import random
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="Diet & Food Recommendation API")

# --------------------------------
# 1. SMART MODEL LOADING
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent

def load_model_safely(filename):
    # Try 1: Look in Root
    path1 = os.path.join(BASE_DIR, filename)
    # Try 2: Look in lib/models
    path2 = os.path.join(BASE_DIR, "lib", "models", filename)
    
    if os.path.exists(path1):
        return joblib.load(path1)
    elif os.path.exists(path2):
        return joblib.load(path2)
    else:
        # This will show you exactly what files ARE there in the logs
        files_present = os.listdir(BASE_DIR)
        raise FileNotFoundError(f"Could not find {filename}. Files found in root: {files_present}")

# Load models
try:
    diet_model = load_model_safely("diet_model.joblib")
    food_model = load_model_safely("dish_recommender_model_extended.pkl")
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")
    # Don't crash the whole server, but the endpoints will fail
    diet_model = None
    food_model = None

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
# 3. ENDPOINTS
# --------------------------------
@app.post("/predict_diet")
def predict_diet(request: DietRequest):
    if not diet_model: return {"error": "Diet model not loaded"}
    height_m = request.height / 100
    bmi = request.weight / (height_m ** 2)
    X = pd.DataFrame([{"age": request.age, "weight": request.weight, "height": request.height, "bmi": bmi, "gender": request.gender, "goal": request.goal, "diet": request.diet, "disease": request.disease}])
    pred = diet_model.predict(X)[0]
    return {"diet_category": pred}

@app.post("/predict_food")
def predict_food(request: FoodRequest):
    if not food_model: return {"error": "Food model not loaded"}
    X = pd.DataFrame([{"diet_type": request.diet_category, "meal_type": request.meal}])
    try:
        probs = food_model.predict_proba(X)[0]
        classes = food_model.named_steps["classifier"].classes_
        top_indices = np.argsort(probs)[-10:]
        random_selection = random.choice(classes[top_indices])
        clean_name = str(random_selection).replace("_", " ").title()
        return {"recommended_food": clean_name}
    except Exception as e:
        return {"error": str(e)}
