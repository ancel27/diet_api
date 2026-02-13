import os
import pandas as pd
import joblib
import numpy as np
import random
from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------
# 1. LOAD MODELS
# --------------------------------
# Get the current folder path
CURRENT_DIR = os.path.dirname(__file__)

# Define paths
diet_path = os.path.join(CURRENT_DIR, "lib", "models", "diet_model.joblib")
food_path = os.path.join(CURRENT_DIR, "dish_recommender_model_extended.pkl")

# Load Diet Model
try:
    diet_model = joblib.load(diet_path)
except:
    diet_model = None

# Load Food Model
try:
    food_model = joblib.load(food_path)
    print("✅ Food model loaded!")
except Exception as e:
    print(f"❌ Error: {e}")
    food_model = None

app = FastAPI()

# --------------------------------
# 2. SCHEMAS
# --------------------------------
class FoodRequest(BaseModel):
    diet_category: str
    meal: str

class DietRequest(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    goal: str
    diet: str
    disease: str

# --------------------------------
# 3. ENDPOINTS
# --------------------------------

@app.post("/predict_diet")
def predict_diet(request: DietRequest):
    if not diet_model:
        return {"error": "Diet model file missing on server"}
    
    height_m = request.height / 100
    bmi = request.weight / (height_m ** 2)
    
    X = pd.DataFrame([{
        "age": request.age, "weight": request.weight, "height": request.height,
        "bmi": bmi, "gender": request.gender, "goal": request.goal,
        "diet": request.diet, "disease": request.disease
    }])
    
    return {"diet_category": diet_model.predict(X)[0]}

@app.post("/predict_food")
def predict_food(request: FoodRequest):
    if not food_model:
        return {"error": "Food model file missing on server"}

    # Map input to model columns
    X = pd.DataFrame([{
        "diet_type": request.diet_category,
        "meal_type": request.meal
    }])

    try:
        # Get top 10 likely dishes
        probs = food_model.predict_proba(X)[0]
        classes = food_model.named_steps["classifier"].classes_
        top_indices = np.argsort(probs)[-10:]
        
        # Pick one at random
        pick = random.choice(classes[top_indices])
        
        # Format "chicken_stirfry" -> "Chicken Stirfry"
        formatted_name = str(pick).replace("_", " ").title()
        
        return {"recommended_food": formatted_name}
    except Exception as e:
        return {"error": str(e)}
