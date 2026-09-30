import os
import traceback
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from utils import BASE_DIR, get_df, DATA_PATH
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__, template_folder='templates', static_folder='static')

from routes.data import data_bp
from routes.model import model_bp
from routes.advanced import adv_bp

app.register_blueprint(data_bp)
app.register_blueprint(model_bp)
app.register_blueprint(adv_bp)

model_pipeline = None

def train_model():
    global model_pipeline
    from utils import get_df
    from sklearn.metrics import r2_score
    df = get_df()
    if df is None:
        print("[WARN] No data to train model")
        return
    FEATURES = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']
    TARGET = 'Total Dispatched Trips'
    clean = df[FEATURES + [TARGET]].dropna()
    if len(clean) == 0: return
    X = clean[FEATURES].values
    y = clean[TARGET].values
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gbm', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
    ])
    model_pipeline.fit(X, y)
    r2 = r2_score(y, model_pipeline.predict(X))
    print(f"[OK] GBM trained  R2={r2:.4f}")


# -- train GBM once at startup ----------------------

get_df()
train_model()

# -- static files (landing page) ----------------------

# -- LOGIN ----------------------

# -- PREDICT API ----------------------

if __name__ == '__main__':
    print('\n  RideRush ML Server')
    print('  Landing   : http://127.0.0.1:5000')
    print('  Dashboard : http://127.0.0.1:5000/load-data\n')
    app.run(host='127.0.0.1', port=5000, debug=False)
