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
from routes.load_data import load_data_bp
from routes.eda import eda_bp
from routes.graphs import graphs_bp
from routes.feature_engg import feature_engg_bp
from routes.regression import regression_bp
from routes.trees import trees_bp
from routes.clustering import clustering_bp
from routes.dbscan import dbscan_bp
from routes.pca import pca_bp
from routes.anomaly import anomaly_bp
from routes.tsne_umap import tsne_umap_bp
from routes.data_leakage import data_leakage_bp
from routes.evaluation import evaluation_bp
app.register_blueprint(load_data_bp)
app.register_blueprint(eda_bp)
app.register_blueprint(graphs_bp)
app.register_blueprint(feature_engg_bp)
app.register_blueprint(regression_bp)
app.register_blueprint(trees_bp)
app.register_blueprint(clustering_bp)
app.register_blueprint(dbscan_bp)
app.register_blueprint(pca_bp)
app.register_blueprint(anomaly_bp)
app.register_blueprint(tsne_umap_bp)
app.register_blueprint(data_leakage_bp)
app.register_blueprint(evaluation_bp)





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

# ── static files (landing page) ─────────────────────────
@app.route('/')
def index():      return send_from_directory(BASE_DIR, 'index.html')
@app.route('/style.css')
def css():        return send_from_directory(BASE_DIR, 'style.css', mimetype='text/css')
@app.route('/main.js')
def js():         return send_from_directory(BASE_DIR, 'main.js',  mimetype='application/javascript')
@app.route('/assets/<path:fn>')
def assets(fn):   return send_from_directory(os.path.join(BASE_DIR, 'assets'), fn)


# ── LOGIN ─────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        # Demo: accept any non-empty credentials, redirect to dashboard
        if email and password:
            from flask import redirect
            return redirect('/load-data')
        error = 'Please enter both email and password.'
    return render_template('login.html', error=error)



# ── PREDICT API ─────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model not ready.'}), 500
    try:
        data  = request.get_json(force=True)
        FEATS = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']
        Xp    = pd.DataFrame([[float(data.get('year',2026)), float(data.get('month',1)),
                                float(data.get('vehicles',1)), float(data.get('shared',0))]], columns=FEATS)
        pred  = float(model_pipeline.predict(Xp)[0])
        bt    = str(data.get('baseType', 'medium'))
        final = max(10, round(pred * {'small':.85,'medium':1.0,'large':1.25,'enterprise':1.5}.get(bt, 1.0)))
        ep    = {'enterprise':.15, 'large':.18}.get(bt, .22)
        conf  = min(97, round(60 + np.log10(float(data.get('vehicles',1)) + 1) * 18))
        return jsonify({'predicted': final, 'low': round(final*(1-ep)), 'high': round(final*(1+ep)), 'confidence': conf})
    except Exception as e:
        traceback.print_exc(); return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print('\n  RideRush ML Server')
    print('  Landing   : http://127.0.0.1:5000')
    print('  Dashboard : http://127.0.0.1:5000/load-data\n')
    app.run(host='127.0.0.1', port=5000, debug=False)
