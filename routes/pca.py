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


pca_bp = Blueprint('pca_bp', __name__)

@pca_bp.route('/pca')
def pca_page():
    df = get_df()
    if df is None:
        return render_template('pca.html', error='Dataset not found.', plots={}, metrics={})
    FEATURES = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips', 'Total Dispatched Trips']
    clean = df[FEATURES].dropna()
    clean = clean[clean['Total Dispatched Trips'] > 0]
    Xs = StandardScaler().fit_transform(clean[FEATURES].values)
    pca_full = PCA(n_components=len(FEATURES)); pca_full.fit(Xs)
    ev_ratio = pca_full.explained_variance_ratio_; cum_ev = np.cumsum(ev_ratio)
    plots = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(FEATURES)+1), ev_ratio*100, color=ORNG, alpha=0.85, edgecolor='#f8fafc')
    axes[0].plot(range(1, len(FEATURES)+1), cum_ev*100, 'o-', color=BLUE, linewidth=2, label='Cumulative')
    axes[0].axhline(y=90, color=RED, linestyle='--', linewidth=1.5, label='90% threshold')
    axes[0].set_xlabel('Principal Component'); axes[0].set_ylabel('Explained Variance (%)')
    axes[0].set_title('Scree Plot - PCA Explained Variance', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.5)
    pca2 = PCA(n_components=2); X2 = pca2.fit_transform(Xs)
    axes[1].scatter(X2[:, 0], X2[:, 1], alpha=0.4, s=8, color=BLUE)
    for i, feat in enumerate(FEATURES):
        label = feat.replace('Total Dispatched ', '').replace('Unique Dispatched ', '')
        cx, cy = pca2.components_[0, i]*3, pca2.components_[1, i]*3
        axes[1].annotate('', xy=(cx, cy), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=ORNG, lw=1.5))
        axes[1].text(cx*1.3, cy*1.3, label, fontsize=7, color=ORNG, fontweight='bold', ha='center')
    axes[1].set_xlabel(f'PC1 ({ev_ratio[0]*100:.1f}%)'); axes[1].set_ylabel(f'PC2 ({ev_ratio[1]*100:.1f}%)')
    axes[1].set_title('PCA Biplot (PC1 vs PC2)', fontsize=11, fontweight='bold'); axes[1].grid(True, alpha=0.5)
    plt.tight_layout(); plots['scree_biplot'] = fig_b64(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    loadings = pca_full.components_
    short_feats = [f.replace('Total Dispatched ', 'TD ').replace('Unique Dispatched ', 'UD ') for f in FEATURES]
    im = ax.imshow(loadings, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(short_feats, rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels([f'PC{i+1}' for i in range(len(FEATURES))], fontsize=9)
    ax.set_title('PCA Component Loadings Heatmap', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f'{loadings[i,j]:.2f}', ha='center', va='center', fontsize=7)
    plt.tight_layout(); plots['loadings_heatmap'] = fig_b64(fig)
    return render_template('pca.html', error=None, plots=plots, metrics={
        'n_samples': len(clean), 'n_features': len(FEATURES), 'features': FEATURES,
        'ev_ratio': [round(v*100, 2) for v in ev_ratio],
        'cum_ev': [round(v*100, 2) for v in cum_ev],
        'n_components_90': int(np.argmax(cum_ev >= 0.90)) + 1
    })
