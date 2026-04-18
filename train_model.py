"""
Model Training Pipeline for Farmer's Crop Disease Prediction
Trains multiple ML models and saves the best one with preprocessing pipeline
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'temperature_celsius', 'humidity_percent', 'rainfall_mm', 'soil_ph',
    'nitrogen_kg_ha', 'phosphorus_kg_ha', 'potassium_kg_ha',
    'wind_speed_kmh', 'sunlight_hours', 'leaf_wetness_hours',
    'field_area_hectares', 'days_after_sowing', 'prev_disease_history'
]

TARGET_COL = 'disease_label'

def train_model(data_path="data/crop_disease_dataset.csv", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("FARMER'S DISEASE PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\n✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Prepare features and target
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save label encoder
    joblib.dump(le, f"{save_dir}/label_encoder.pkl")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"✅ Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Build pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("\n🔄 Training Random Forest model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 MODEL PERFORMANCE METRICS:")
    print(f"   Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    
    # Cross validation
    cv_scores = cross_val_score(pipeline, X, y_encoded, cv=5, scoring='accuracy')
    print(f"\n📈 5-Fold Cross Validation:")
    print(f"   CV Scores : {cv_scores}")
    print(f"   Mean±Std  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    clf = pipeline.named_steps['classifier']
    feature_importance = dict(zip(FEATURE_COLS, clf.feature_importances_.tolist()))
    feature_importance_sorted = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\n🌟 Top 5 Important Features:")
    for i, (feat, imp) in enumerate(list(feature_importance_sorted.items())[:5]):
        print(f"   {i+1}. {feat}: {imp:.4f}")
    
    # Class report
    class_names = le.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Save all artifacts
    joblib.dump(pipeline, f"{save_dir}/disease_model.pkl")
    
    # Save metadata
    metadata = {
        "model_type": "Random Forest Classifier",
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "cv_scores": cv_scores.tolist(),
        "feature_columns": FEATURE_COLS,
        "target_column": TARGET_COL,
        "classes": class_names.tolist(),
        "n_estimators": 200,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_importance": feature_importance_sorted,
        "classification_report": report
    }
    
    with open(f"{save_dir}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to {save_dir}/disease_model.pkl")
    print(f"✅ Metadata saved to {save_dir}/model_metadata.json")
    print(f"✅ Label encoder saved to {save_dir}/label_encoder.pkl")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return pipeline, le, metadata

if __name__ == "__main__":
    train_model()
