from flask import Blueprint, render_template, request, jsonify, redirect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_df, fig_b64, ORNG, BLUE, GRN, PURP, RED
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE


trees_bp = Blueprint('trees_bp', __name__)

@trees_bp.route('/trees')
def trees_page():
    df = get_df()
    if df is None:
        return render_template('trees.html', error='Dataset not found.', tree={}, plots={})
    
    TARGET = 'Total Dispatched Trips'
    SLRF   = 'Unique Dispatched Vehicles'
    MLRF   = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']
    plots  = {}

    clean = df[list(set([TARGET, SLRF] + MLRF))].dropna()
    clean = clean[clean[TARGET] > 0]
    y     = clean[TARGET].values

    Xm = clean[MLRF].values
    Xtr, Xte, ytr, yte = train_test_split(Xm, y, test_size=.2, random_state=42)
    
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(Xtr, ytr)
    yp = dt.predict(Xte)

    tree_metrics = dict(
        features=MLRF, target=TARGET,
        r2=round(r2_score(yte, yp), 4),
        mse=round(mean_squared_error(yte, yp), 2),
        mae=round(mean_absolute_error(yte, yp), 2),
        rmse=round(float(np.sqrt(mean_squared_error(yte, yp))), 2),
        n_train=len(Xtr), n_test=len(Xte)
    )

    # Plot the tree
    fig, ax = plt.subplots(figsize=(20, 6))
    texts = plot_tree(dt, feature_names=MLRF, filled=True, ax=ax, fontsize=7, rounded=True)
    for text in texts:
        text.set_color('black')
    ax.set_title('Decision Tree Regressor (max_depth=3)', fontsize=12, fontweight='bold', color='#0f172a')
    plots['tree_plot'] = fig_b64(fig)

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    mn = min(yte.min(), yp.min()); mx = max(yte.max(), yp.max())
    ax.scatter(yte, yp, alpha=0.7, s=6, color=BLUE)
    ax.plot([mn, mx], [mn, mx], color=ORNG, linewidth=2, linestyle='--', label='Perfect')
    ax.set_title('Decision Tree: Actual vs Predicted', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.7)
    plots['scatter'] = fig_b64(fig)

    # Feature Importances
    fig, ax = plt.subplots(figsize=(7, 5))
    fi = dt.feature_importances_
    fn = [f.replace('Total Dispatched ','').replace('Unique Dispatched ','') for f in MLRF]
    ax.barh(fn, fi, color=ORNG, edgecolor='#f8fafc', alpha=.85)
    ax.set_title('Feature Importances', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.7); plt.tight_layout()
    plots['importances'] = fig_b64(fig)

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(Xtr, ytr)
    yp_rf = rf.predict(Xte)

    rf_metrics = dict(
        features=MLRF, target=TARGET,
        r2=round(r2_score(yte, yp_rf), 4),
        mse=round(mean_squared_error(yte, yp_rf), 2),
        mae=round(mean_absolute_error(yte, yp_rf), 2),
        rmse=round(float(np.sqrt(mean_squared_error(yte, yp_rf))), 2),
        n_train=len(Xtr), n_test=len(Xte)
    )

    # Plot actual vs predicted for RF
    fig, ax = plt.subplots(figsize=(7, 5))
    mn = min(yte.min(), yp_rf.min()); mx = max(yte.max(), yp_rf.max())
    ax.scatter(yte, yp_rf, alpha=0.7, s=6, color=BLUE)
    ax.plot([mn, mx], [mn, mx], color=ORNG, linewidth=2, linestyle='--', label='Perfect')
    ax.set_title('Random Forest: Actual vs Predicted', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.7)
    plots['rf_scatter'] = fig_b64(fig)

    # Feature Importances for RF
    fig, ax = plt.subplots(figsize=(7, 5))
    fi_rf = rf.feature_importances_
    ax.barh(fn, fi_rf, color=GRN, edgecolor='#f8fafc', alpha=.85)
    ax.set_title('RF Feature Importances', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.7); plt.tight_layout()
    plots['rf_importances'] = fig_b64(fig)

    # Gradient Boosting Regressor
    gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    gb.fit(Xtr, ytr)
    yp_gb = gb.predict(Xte)

    gb_metrics = dict(
        features=MLRF, target=TARGET,
        r2=round(r2_score(yte, yp_gb), 4),
        mse=round(mean_squared_error(yte, yp_gb), 2),
        mae=round(mean_absolute_error(yte, yp_gb), 2),
        rmse=round(float(np.sqrt(mean_squared_error(yte, yp_gb))), 2),
        n_train=len(Xtr), n_test=len(Xte)
    )

    # Plot actual vs predicted for GB
    fig, ax = plt.subplots(figsize=(7, 5))
    mn = min(yte.min(), yp_gb.min()); mx = max(yte.max(), yp_gb.max())
    ax.scatter(yte, yp_gb, alpha=0.7, s=6, color=BLUE)
    ax.plot([mn, mx], [mn, mx], color=ORNG, linewidth=2, linestyle='--', label='Perfect')
    ax.set_title('Gradient Boosting: Actual vs Predicted', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.7)
    plots['gb_scatter'] = fig_b64(fig)

    # Feature Importances for GB
    fig, ax = plt.subplots(figsize=(7, 5))
    fi_gb = gb.feature_importances_
    ax.barh(fn, fi_gb, color=PURP, edgecolor='#f8fafc', alpha=.85)
    ax.set_title('GB Feature Importances', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.7); plt.tight_layout()
    plots['gb_importances'] = fig_b64(fig)

    # AdaBoost Regressor
    ab = AdaBoostRegressor(n_estimators=50, random_state=42)
    ab.fit(Xtr, ytr)
    yp_ab = ab.predict(Xte)

    ab_metrics = dict(
        features=MLRF, target=TARGET,
        r2=round(r2_score(yte, yp_ab), 4),
        mse=round(mean_squared_error(yte, yp_ab), 2),
        mae=round(mean_absolute_error(yte, yp_ab), 2),
        rmse=round(float(np.sqrt(mean_squared_error(yte, yp_ab))), 2),
        n_train=len(Xtr), n_test=len(Xte)
    )

    # Plot actual vs predicted for AB
    fig, ax = plt.subplots(figsize=(7, 5))
    mn = min(yte.min(), yp_ab.min()); mx = max(yte.max(), yp_ab.max())
    ax.scatter(yte, yp_ab, alpha=0.7, s=6, color=BLUE)
    ax.plot([mn, mx], [mn, mx], color=ORNG, linewidth=2, linestyle='--', label='Perfect')
    ax.set_title('AdaBoost: Actual vs Predicted', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.7)
    plots['ab_scatter'] = fig_b64(fig)

    # Feature Importances for AB
    fig, ax = plt.subplots(figsize=(7, 5))
    fi_ab = ab.feature_importances_
    ax.barh(fn, fi_ab, color=RED, edgecolor='#f8fafc', alpha=.85)
    ax.set_title('AdaBoost Feature Importances', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.7); plt.tight_layout()
    plots['ab_importances'] = fig_b64(fig)

    return render_template('trees.html', error=None, tree=tree_metrics, rf=rf_metrics, gb=gb_metrics, ab=ab_metrics, plots=plots)
