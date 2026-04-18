"""
Dataset Generator for Farmer's Crop Disease Prediction
Generates a realistic synthetic dataset based on agronomic research parameters
"""

import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Disease categories with realistic agronomic profiles
DISEASES = {
    "Healthy": {
        "temp_range": (20, 30), "humidity_range": (40, 65),
        "rainfall_range": (50, 150), "soil_ph_range": (6.0, 7.5),
        "nitrogen_range": (60, 100), "phosphorus_range": (40, 80),
        "potassium_range": (50, 90), "weight": 0.25
    },
    "Leaf Blight": {
        "temp_range": (25, 35), "humidity_range": (70, 95),
        "rainfall_range": (100, 250), "soil_ph_range": (5.5, 7.0),
        "nitrogen_range": (30, 70), "phosphorus_range": (20, 55),
        "potassium_range": (30, 65), "weight": 0.12
    },
    "Powdery Mildew": {
        "temp_range": (18, 28), "humidity_range": (45, 75),
        "rainfall_range": (20, 80), "soil_ph_range": (6.0, 8.0),
        "nitrogen_range": (40, 80), "phosphorus_range": (30, 65),
        "potassium_range": (40, 75), "weight": 0.10
    },
    "Root Rot": {
        "temp_range": (15, 25), "humidity_range": (75, 100),
        "rainfall_range": (150, 350), "soil_ph_range": (4.5, 6.5),
        "nitrogen_range": (20, 55), "phosphorus_range": (15, 45),
        "potassium_range": (20, 50), "weight": 0.10
    },
    "Rust Disease": {
        "temp_range": (15, 25), "humidity_range": (60, 90),
        "rainfall_range": (60, 180), "soil_ph_range": (5.5, 7.0),
        "nitrogen_range": (35, 70), "phosphorus_range": (25, 60),
        "potassium_range": (35, 70), "weight": 0.10
    },
    "Bacterial Wilt": {
        "temp_range": (25, 38), "humidity_range": (65, 90),
        "rainfall_range": (80, 200), "soil_ph_range": (5.0, 7.5),
        "nitrogen_range": (25, 65), "phosphorus_range": (20, 55),
        "potassium_range": (25, 60), "weight": 0.08
    },
    "Downy Mildew": {
        "temp_range": (10, 22), "humidity_range": (80, 100),
        "rainfall_range": (100, 300), "soil_ph_range": (5.5, 7.5),
        "nitrogen_range": (30, 65), "phosphorus_range": (20, 50),
        "potassium_range": (30, 60), "weight": 0.08
    },
    "Mosaic Virus": {
        "temp_range": (22, 32), "humidity_range": (50, 80),
        "rainfall_range": (40, 150), "soil_ph_range": (6.0, 7.5),
        "nitrogen_range": (35, 75), "phosphorus_range": (25, 60),
        "potassium_range": (35, 70), "weight": 0.07
    },
    "Anthracnose": {
        "temp_range": (20, 30), "humidity_range": (75, 100),
        "rainfall_range": (120, 280), "soil_ph_range": (5.0, 7.0),
        "nitrogen_range": (30, 65), "phosphorus_range": (20, 55),
        "potassium_range": (30, 65), "weight": 0.06
    },
    "Fusarium Wilt": {
        "temp_range": (24, 34), "humidity_range": (55, 85),
        "rainfall_range": (60, 180), "soil_ph_range": (4.5, 6.5),
        "nitrogen_range": (25, 60), "phosphorus_range": (15, 50),
        "potassium_range": (25, 55), "weight": 0.04
    },
}

CROPS = ["Wheat", "Rice", "Maize", "Tomato", "Potato", "Cotton", "Soybean", "Sugarcane", "Barley", "Sorghum"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
SOIL_TYPES = ["Sandy Loam", "Clay", "Loam", "Sandy", "Silty Clay", "Black Cotton", "Red Laterite"]
STATES = [
    "Uttar Pradesh", "Punjab", "Haryana", "Maharashtra", "Madhya Pradesh",
    "Rajasthan", "Gujarat", "Bihar", "West Bengal", "Andhra Pradesh",
    "Karnataka", "Tamil Nadu", "Telangana", "Odisha", "Jharkhand"
]

def generate_sample(disease_name, disease_info, sample_id):
    t_min, t_max = disease_info["temp_range"]
    h_min, h_max = disease_info["humidity_range"]
    r_min, r_max = disease_info["rainfall_range"]
    p_min, p_max = disease_info["soil_ph_range"]
    n_min, n_max = disease_info["nitrogen_range"]
    ph_min, ph_max = disease_info["phosphorus_range"]
    k_min, k_max = disease_info["potassium_range"]

    temperature = round(np.random.normal((t_min+t_max)/2, (t_max-t_min)/6), 1)
    temperature = max(t_min-3, min(t_max+3, temperature))

    humidity = round(np.random.normal((h_min+h_max)/2, (h_max-h_min)/6), 1)
    humidity = max(max(0, h_min-5), min(100, humidity))

    rainfall = round(np.random.normal((r_min+r_max)/2, (r_max-r_min)/6), 2)
    rainfall = max(0, rainfall)

    soil_ph = round(np.random.normal((p_min+p_max)/2, (p_max-p_min)/6), 2)
    soil_ph = max(3.5, min(9.0, soil_ph))

    nitrogen = round(np.random.normal((n_min+n_max)/2, (n_max-n_min)/6), 1)
    phosphorus = round(np.random.normal((ph_min+ph_max)/2, (ph_max-ph_min)/6), 1)
    potassium = round(np.random.normal((k_min+k_max)/2, (k_max-k_min)/6), 1)

    nitrogen = max(0, nitrogen)
    phosphorus = max(0, phosphorus)
    potassium = max(0, potassium)

    crop = random.choice(CROPS)
    season = random.choice(SEASONS)
    soil_type = random.choice(SOIL_TYPES)
    state = random.choice(STATES)
    
    # Wind speed affects disease spread
    wind_speed = round(np.random.uniform(2, 25), 1)
    
    # Sunlight hours
    sunlight_hours = round(np.random.uniform(4, 12), 1)
    
    # Leaf wetness (hours)
    leaf_wetness = round(np.random.uniform(1, 18), 1)
    
    # Field area
    field_area_hectares = round(np.random.uniform(0.5, 25), 2)
    
    # Days after sowing
    days_after_sowing = random.randint(15, 120)
    
    # Previous crop disease history
    prev_disease_history = random.choice([0, 1])
    
    # Irrigation type
    irrigation = random.choice(["Drip", "Flood", "Sprinkler", "Rainfed", "Furrow"])
    
    # Severity score (for research analysis)
    if disease_name == "Healthy":
        severity_score = random.uniform(0, 0.1)
        severity_label = "None"
        treatment_cost_inr = 0
        yield_loss_percent = 0
    else:
        severity_score = random.uniform(0.3, 1.0)
        if severity_score < 0.5:
            severity_label = "Mild"
            treatment_cost_inr = random.randint(500, 2000)
            yield_loss_percent = round(random.uniform(5, 20), 1)
        elif severity_score < 0.75:
            severity_label = "Moderate"
            treatment_cost_inr = random.randint(2000, 8000)
            yield_loss_percent = round(random.uniform(20, 45), 1)
        else:
            severity_label = "Severe"
            treatment_cost_inr = random.randint(8000, 25000)
            yield_loss_percent = round(random.uniform(45, 80), 1)
    
    return {
        "sample_id": f"FDP-{sample_id:05d}",
        "crop_type": crop,
        "state": state,
        "season": season,
        "soil_type": soil_type,
        "temperature_celsius": temperature,
        "humidity_percent": humidity,
        "rainfall_mm": rainfall,
        "soil_ph": soil_ph,
        "nitrogen_kg_ha": nitrogen,
        "phosphorus_kg_ha": phosphorus,
        "potassium_kg_ha": potassium,
        "wind_speed_kmh": wind_speed,
        "sunlight_hours": sunlight_hours,
        "leaf_wetness_hours": leaf_wetness,
        "field_area_hectares": field_area_hectares,
        "days_after_sowing": days_after_sowing,
        "irrigation_type": irrigation,
        "prev_disease_history": prev_disease_history,
        "disease_label": disease_name,
        "severity_score": round(severity_score, 3),
        "severity_label": severity_label,
        "treatment_cost_inr": treatment_cost_inr,
        "yield_loss_percent": yield_loss_percent
    }

def generate_dataset(n_samples=5000):
    diseases = list(DISEASES.keys())
    weights = [DISEASES[d]["weight"] for d in diseases]
    
    records = []
    for i in range(1, n_samples + 1):
        chosen_disease = random.choices(diseases, weights=weights, k=1)[0]
        record = generate_sample(chosen_disease, DISEASES[chosen_disease], i)
        records.append(record)
    
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset(5000)
    df.to_csv("crop_disease_dataset.csv", index=False)
    print(f"Dataset generated: {len(df)} samples")
    print(f"Disease distribution:\n{df['disease_label'].value_counts()}")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
