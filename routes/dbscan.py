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


dbscan_bp = Blueprint('dbscan_bp', __name__)

@dbscan_bp.route('/dbscan')
def dbscan_page():
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    df = get_df()
    if df is None:
        return render_template('dbscan.html', error='Dataset not found.', plots={}, metrics={})
    FEATURES = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips', 'Total Dispatched Trips']
    clean = df[FEATURES].dropna()
    clean = clean[clean['Total Dispatched Trips'] > 0].sample(n=min(len(clean), 2000), random_state=42)
    X = clean[FEATURES].values
    Xs = StandardScaler().fit_transform(X)
    X2 = PCA(n_components=2).fit_transform(Xs)
    plots = {}
    results = []
    for eps in [0.3, 0.5, 0.8, 1.2]:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(Xs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        try:
            sil = round(silhouette_score(Xs, labels), 4) if n_clusters > 1 else 'N/A'
        except Exception:
            sil = 'N/A'
        results.append({'eps': eps, 'clusters': n_clusters, 'noise': n_noise, 'silhouette': sil})
    db_best = DBSCAN(eps=0.5, min_samples=5)
    best_labels = db_best.fit_predict(Xs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(X2[:, 0], X2[:, 1], c=best_labels, cmap='tab10', alpha=0.7, s=15)
    axes[0].set_title('DBSCAN Clusters (eps=0.5) - PCA 2D', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('PCA 1'); axes[0].set_ylabel('PCA 2'); axes[0].grid(True, alpha=0.5)
    noise_mask = best_labels == -1
    axes[1].scatter(X2[~noise_mask, 0], X2[~noise_mask, 1], c=best_labels[~noise_mask], cmap='tab10', alpha=0.7, s=15, label='Cluster')
    axes[1].scatter(X2[noise_mask, 0], X2[noise_mask, 1], c='red', alpha=0.5, s=10, marker='x', label='Noise')
    axes[1].set_title('DBSCAN: Core vs Noise Points', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('PCA 1'); axes[1].set_ylabel('PCA 2')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.5)
    plt.tight_layout(); plots['dbscan_main'] = fig_b64(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    eps_vals = [r['eps'] for r in results]
    ax.plot(eps_vals, [r['clusters'] for r in results], 'o-', color=ORNG, linewidth=2, label='Clusters', markersize=8)
    ax2b = ax.twinx()
    ax2b.plot(eps_vals, [r['noise'] for r in results], 's--', color=RED, linewidth=2, label='Noise pts', markersize=8)
    ax.set_xlabel('eps value'); ax.set_ylabel('# Clusters', color=ORNG); ax2b.set_ylabel('# Noise Points', color=RED)
    ax.set_title('DBSCAN: eps Sensitivity Analysis', fontsize=11, fontweight='bold'); ax.grid(True, alpha=0.5)
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2b.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8)
    plt.tight_layout(); plots['eps_sensitivity'] = fig_b64(fig)
    return render_template('dbscan.html', error=None,
                           metrics={'n_samples': len(Xs), 'n_features': len(FEATURES), 'features': FEATURES, 'results': results},
                           plots=plots)
