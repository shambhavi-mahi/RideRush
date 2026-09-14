import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import traceback

# ── App setup ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = r"C:\Users\SHAMBHAVI\Downloads\FHV_Base_Aggregate_Report_20260812.csv"

app = Flask(__name__)
model_pipeline = None


# ── ML: Data Ingestion, Cleaning, Training ────────────────────────────────
def load_and_train_model():
    global model_pipeline
    print("Loading FHV dataset...")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)

        # Data Cleaning (strip commas from numbers)
        def parse_num(x):
            if pd.isna(x):
                return 0.0
            return float(str(x).replace(',', '').strip()) if isinstance(x, str) else float(x)

        df['Year']                          = df['Year'].apply(parse_num)
        df['Month']                         = df['Month'].apply(parse_num)
        df['Total Dispatched Trips']        = df['Total Dispatched Trips'].apply(parse_num)
        df['Total Dispatched Shared Trips'] = df['Total Dispatched Shared Trips'].apply(parse_num)
        df['Unique Dispatched Vehicles']    = df['Unique Dispatched Vehicles'].apply(parse_num)

        df = df[df['Total Dispatched Trips'] > 0].dropna(subset=['Total Dispatched Trips'])

        # Feature Engineering
        X = df[['Year', 'Month', 'Unique Dispatched Vehicles',
                 'Total Dispatched Shared Trips']].copy()
        y = df['Total Dispatched Trips']

        # Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Pipeline: Imputer -> StandardScaler -> GradientBoosting
        print("Training Gradient Boosting model...")
        model_pipeline = Pipeline([
            ('imputer',   SimpleImputer(strategy='median')),
            ('scaler',    StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.1,
                max_depth=4, random_state=42)),
        ])
        model_pipeline.fit(X_train, y_train)
        score = model_pipeline.score(X_test, y_test)
        print(f"[OK] Model ready  |  R2 = {score:.4f}")

    except Exception:
        print("[ERROR] Model training failed:")
        traceback.print_exc()


load_and_train_model()


# ── Static File Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/style.css')
def stylesheet():
    return send_from_directory(BASE_DIR, 'style.css', mimetype='text/css')

@app.route('/main.js')
def javascript():
    return send_from_directory(BASE_DIR, 'main.js', mimetype='application/javascript')

@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'assets'), filename)


# ── ML Prediction API ──────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model not ready. Check that the dataset exists.'}), 500

    try:
        data     = request.get_json(force=True)
        month    = float(data.get('month',    1))
        year     = float(data.get('year',     2026))
        vehicles = float(data.get('vehicles', 1))
        shared   = float(data.get('shared',   0))
        baseType = str(data.get('baseType', 'medium'))

        X_pred = pd.DataFrame(
            [[year, month, vehicles, shared]],
            columns=['Year', 'Month', 'Unique Dispatched Vehicles',
                     'Total Dispatched Shared Trips']
        )

        pred_val = float(model_pipeline.predict(X_pred)[0])

        # Base-type organisational scale multiplier
        base_weights = {'small': 0.85, 'medium': 1.0, 'large': 1.25, 'enterprise': 1.5}
        weight = base_weights.get(baseType, 1.0)
        final  = max(10, round(pred_val * weight))

        # Confidence interval
        err_pct    = {'enterprise': 0.15, 'large': 0.18}.get(baseType, 0.22)
        confidence = min(97, round(60 + np.log10(vehicles + 1) * 18))

        return jsonify({
            'predicted':  final,
            'low':        round(final * (1 - err_pct)),
            'high':       round(final * (1 + err_pct)),
            'confidence': confidence,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  RideRush ML Server")
    print("  Open http://127.0.0.1:5000  in your browser\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
