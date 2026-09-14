import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS so the frontend can communicate with the backend

DATA_PATH = r"C:\Users\SHAMBHAVI\Downloads\FHV_Base_Aggregate_Report_20260812.csv"
model_pipeline = None

def load_and_train_model():
    """
    ML Features ported from Placement Predict project:
    - Data Ingestion & Cleaning
    - Train/Test Split
    - Feature Scaling (StandardScaler)
    - Regression/Tree modeling (GradientBoostingRegressor)
    """
    global model_pipeline
    print("Loading FHV dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}")
        return
        
    try:
        df = pd.read_csv(DATA_PATH)
        
        # --- Data Cleaning ---
        def parse_num(x):
            if pd.isna(x): return 0.0
            if isinstance(x, str):
                return float(x.replace(',', '').strip())
            return float(x)
            
        df['Year'] = df['Year'].apply(parse_num)
        df['Month'] = df['Month'].apply(parse_num)
        df['Total Dispatched Trips'] = df['Total Dispatched Trips'].apply(parse_num)
        df['Total Dispatched Shared Trips'] = df['Total Dispatched Shared Trips'].apply(parse_num)
        df['Unique Dispatched Vehicles'] = df['Unique Dispatched Vehicles'].apply(parse_num)
        
        # Drop rows missing the target variable
        df = df.dropna(subset=['Total Dispatched Trips'])
        
        # --- Feature Engineering & Selection ---
        X = df[['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']].copy()
        y = df['Total Dispatched Trips']
        
        # --- Train/Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training ML model pipeline...")
        # --- Model Pipeline (Scaling + Tree Regressor) ---
        model_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        model_pipeline.fit(X_train, y_train)
        score = model_pipeline.score(X_test, y_test)
        print(f"✅ Model trained successfully! (Test R² Score: {score:.4f})")
        
    except Exception as e:
        print("Error during model training:")
        traceback.print_exc()

# Initialize model at startup
load_and_train_model()

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model not trained or data not found on server.'}), 500
        
    try:
        data = request.json
        month = float(data.get('month', 1))
        year = float(data.get('year', 2026))
        vehicles = float(data.get('vehicles', 1))
        shared = float(data.get('shared', 0))
        baseType = data.get('baseType', 'medium')
        
        # Create input array matching the training features
        X_pred = pd.DataFrame([[year, month, vehicles, shared]], 
                              columns=['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips'])
        
        # Run ML prediction
        pred_val = model_pipeline.predict(X_pred)[0]
        
        # Base type adjustment to simulate organizational scale impacts not present in raw base data
        base_weights = {'small': 0.85, 'medium': 1.0, 'large': 1.25, 'enterprise': 1.5}
        weight = base_weights.get(baseType, 1.0)
        
        final_prediction = max(10, round(pred_val * weight))
        
        # Generate frontend stats
        errPct = 0.15 if baseType == 'enterprise' else 0.18 if baseType == 'large' else 0.22
        low = round(final_prediction * (1 - errPct))
        high = round(final_prediction * (1 + errPct))
        confidence = min(97, 60 + np.log10(vehicles + 1) * 18)
        
        return jsonify({
            'predicted': final_prediction,
            'low': low,
            'high': high,
            'confidence': round(confidence)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
