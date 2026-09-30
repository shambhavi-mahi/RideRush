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


anomaly_bp = Blueprint('anomaly_bp', __name__)

@anomaly_bp.route('/anomaly')
def anomaly_page():
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from matplotlib.patches import Patch
    df = get_df()
    if df is None:
        return render_template('anomaly.html', error='Dataset not found.', plots={}, metrics={})
    FEATURES = ['Unique Dispatched Vehicles', 'Total Dispatched Shared Trips', 'Total Dispatched Trips']
    clean = df[FEATURES].dropna()
    clean = clean[clean['Total Dispatched Trips'] > 0].sample(n=min(len(clean), 2000), random_state=42)
    Xs = StandardScaler().fit_transform(clean[FEATURES].values)
    X2 = PCA(n_components=2).fit_transform(Xs)
    plots = {}
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_labels = iso.fit_predict(Xs); iso_scores = iso.decision_function(Xs)
    iso_anomalies = (iso_labels == -1).sum()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_labels = lof.fit_predict(Xs); lof_anomalies = (lof_labels == -1).sum()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(X2[:, 0], X2[:, 1], c=['#EF4444' if l==-1 else '#3B82F6' for l in iso_labels], alpha=0.6, s=12)
    axes[0].set_title(f'Isolation Forest ({iso_anomalies} anomalies)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('PCA 1'); axes[0].set_ylabel('PCA 2')
    axes[0].legend(handles=[Patch(color='#EF4444', label='Anomaly'), Patch(color='#3B82F6', label='Normal')], fontsize=8)
    axes[0].grid(True, alpha=0.5)
    axes[1].scatter(X2[:, 0], X2[:, 1], c=['#EF4444' if l==-1 else '#10B981' for l in lof_labels], alpha=0.6, s=12)
    axes[1].set_title(f'LOF ({lof_anomalies} anomalies)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('PCA 1'); axes[1].set_ylabel('PCA 2')
    axes[1].legend(handles=[Patch(color='#EF4444', label='Anomaly'), Patch(color='#10B981', label='Normal')], fontsize=8)
    axes[1].grid(True, alpha=0.5)
    plt.tight_layout(); plots['anomaly_comparison'] = fig_b64(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(iso_scores, bins=50, color=ORNG, alpha=0.7, edgecolor='white', label='Anomaly scores')
    ax.axvline(x=0, color=RED, linestyle='--', linewidth=2, label='Decision boundary')
    ax.set_xlabel('Anomaly Score'); ax.set_ylabel('Count')
    ax.set_title('Isolation Forest: Score Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.5); plt.tight_layout()
    plots['score_dist'] = fig_b64(fig)
    ts_cols = list(dict.fromkeys(['Year', 'Month', 'Total Dispatched Trips'] + FEATURES))
    ts_clean = df[[c for c in ts_cols if c in df.columns]].dropna()
    ts_clean = ts_clean[ts_clean['Total Dispatched Trips']>0].copy()
    ts_Xs = StandardScaler().fit_transform(ts_clean[FEATURES].values)
    ts_labels = IsolationForest(contamination=0.05, random_state=42).fit_predict(ts_Xs)
    ts_clean['anomaly'] = ts_labels; ts_clean['idx'] = range(len(ts_clean))
    normal = ts_clean[ts_clean['anomaly']==1]; anomal = ts_clean[ts_clean['anomaly']==-1]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts_clean['idx'], ts_clean['Total Dispatched Trips'], color=BLUE, alpha=0.4, linewidth=1)
    ax.scatter(normal['idx'], normal['Total Dispatched Trips'], color=BLUE, alpha=0.5, s=8, label='Normal')
    ax.scatter(anomal['idx'], anomal['Total Dispatched Trips'], color=RED, alpha=0.9, s=40, zorder=5, label='Anomaly', marker='*')
    ax.set_xlabel('Record Index'); ax.set_ylabel('Total Dispatched Trips')
    ax.set_title('Anomaly Detection on Trip Volume', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.5); plt.tight_layout()
    plots['time_series'] = fig_b64(fig)
    return render_template('anomaly.html', error=None, plots=plots, metrics={
        'n_samples': len(clean), 'n_features': len(FEATURES), 'features': FEATURES,
        'iso_anomalies': int(iso_anomalies), 'lof_anomalies': int(lof_anomalies), 'contamination': '5%'
    })
