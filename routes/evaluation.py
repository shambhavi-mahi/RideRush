from flask import Blueprint, render_template
import numpy as np
from utils import get_df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

evaluation_bp = Blueprint('evaluation_bp', __name__)

@evaluation_bp.route('/evaluation')
def evaluation_page():
    df = get_df()
    if df is None:
        return render_template('evaluation.html', error='Dataset not found.')

    FEATURES = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']
    TARGET = 'Total Dispatched Trips'
    
    clean = df[FEATURES + [TARGET]].dropna()
    if len(clean) == 0:
        return render_template('evaluation.html', error='Not enough data.')
        
    X = clean[FEATURES].values
    y = clean[TARGET].values

    # --- Regression Metrics ---
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression().fit(Xtr, ytr)
    yp = lr.predict(Xte)
    
    reg_metrics = {
        'mae': round(mean_absolute_error(yte, yp), 2),
        'mse': round(mean_squared_error(yte, yp), 2),
        'rmse': round(np.sqrt(mean_squared_error(yte, yp)), 2),
        'r2': round(r2_score(yte, yp), 4)
    }

    # --- Classification Metrics ---
    # Create a binary target: 1 if trips > median else 0
    median_trips = np.median(y)
    y_class = (y > median_trips).astype(int)
    
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    ss = StandardScaler()
    Xtr_cs = ss.fit_transform(Xtr_c)
    Xte_cs = ss.transform(Xte_c)
    
    logr = LogisticRegression().fit(Xtr_cs, ytr_c)
    yp_c = logr.predict(Xte_cs)
    
    clf_metrics = {
        'accuracy': round(accuracy_score(yte_c, yp_c), 4),
        'precision': round(precision_score(yte_c, yp_c), 4),
        'recall': round(recall_score(yte_c, yp_c), 4),
        'f1': round(f1_score(yte_c, yp_c), 4),
        'cm': confusion_matrix(yte_c, yp_c).tolist()
    }

    return render_template('evaluation.html', error=None, reg=reg_metrics, clf=clf_metrics)
