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


clustering_bp = Blueprint('clustering_bp', __name__)

@clustering_bp.route('/clustering')
def clustering_page():
    df = get_df()
    if df is None:
        return render_template('clustering.html', error='Dataset not found.', plots={})
    
    # We use numerical features for clustering
    FEATURES = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips', 'Total Dispatched Trips']
    
    clean = df[list(set(FEATURES))].dropna()
    clean = clean[clean['Total Dispatched Trips'] > 0]
    # Downsample for hierarchical clustering performance and visual clarity
    clean = clean.sample(n=min(len(clean), 3000), random_state=42)
    X = clean[FEATURES].values

    # Scale data
    ss = StandardScaler()
    Xs = ss.fit_transform(X)
    
    plots = {}

    # 1. KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(Xs)
    
    # 2. Hierarchical (Agglomerative)
    hc = AgglomerativeClustering(n_clusters=3)
    hc_labels = hc.fit_predict(Xs)
    
    # Dimensionality Reduction for plotting 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xs)
    
    # Plot KMeans
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap='viridis', alpha=0.7, s=20, edgecolor='#f8fafc', linewidth=0.3)
    ax.set_title('K-Means Clustering (3 Clusters) - PCA Projection', fontsize=11, fontweight='bold')
    ax.set_xlabel('PCA Component 1'); ax.set_ylabel('PCA Component 2')
    ax.grid(True, alpha=0.7); plt.tight_layout()
    plots['kmeans_plot'] = fig_b64(fig)
    
    # Plot Hierarchical
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=hc_labels, cmap='plasma', alpha=0.7, s=20, edgecolor='#f8fafc', linewidth=0.3)
    ax.set_title('Hierarchical Clustering (3 Clusters) - PCA Projection', fontsize=11, fontweight='bold')
    ax.set_xlabel('PCA Component 1'); ax.set_ylabel('PCA Component 2')
    ax.grid(True, alpha=0.7); plt.tight_layout()
    plots['hc_plot'] = fig_b64(fig)
    
    metrics = {
        'n_samples': len(Xs),
        'n_features': len(FEATURES),
        'features': FEATURES,
        'k': 3
    }
    
    return render_template('clustering.html', error=None, metrics=metrics, plots=plots)
