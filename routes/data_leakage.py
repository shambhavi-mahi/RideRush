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


data_leakage_bp = Blueprint('data_leakage_bp', __name__)

@data_leakage_bp.route('/data-leakage')
def data_leakage_page():
    from sklearn.linear_model import LinearRegression as LR
    df = get_df()
    if df is None:
        return render_template('data_leakage.html', error='Dataset not found.', plots={}, metrics={})
    TARGET = 'Total Dispatched Trips'
    FEATURES = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']
    ALL_COLS = FEATURES + [TARGET]
    clean = df[ALL_COLS].dropna(); clean = clean[clean[TARGET] > 0]
    corr = clean[ALL_COLS].corr()
    X = clean[FEATURES].values; y = clean[TARGET].values
    plots = {}
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
    short = [c.replace('Total Dispatched ', 'TD ').replace('Unique Dispatched ', 'UD ') for c in ALL_COLS]
    ax.set_xticks(range(len(ALL_COLS))); ax.set_xticklabels(short, rotation=25, ha='right', fontsize=8)
    ax.set_yticks(range(len(ALL_COLS))); ax.set_yticklabels(short, fontsize=8)
    ax.set_title('Feature Correlation Matrix (Leakage Check)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(ALL_COLS)):
        for j in range(len(ALL_COLS)):
            ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(corr.values[i,j]) > 0.6 else 'black')
    plt.tight_layout(); plots['corr_heatmap'] = fig_b64(fig)
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    split_models = {}
    for name, (Xtr, ytr, Xte, yte) in [('Temporal Split', (Xtr_r, ytr_r, Xte_r, yte_r)),
                                          ('Random Split',   (Xtr_s, ytr_s, Xte_s, yte_s))]:
        m = LR().fit(Xtr, ytr); yp = m.predict(Xte)
        split_models[name] = {'r2': round(r2_score(yte, yp), 4), 'mae': round(mean_absolute_error(yte, yp), 2)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(clean['Total Dispatched Shared Trips'], clean[TARGET], alpha=0.4, s=8, color=ORNG)
    axes[0].set_xlabel('Total Dispatched Shared Trips'); axes[0].set_ylabel(TARGET)
    axes[0].set_title('Potential Leakage: Shared vs Total Trips', fontsize=11, fontweight='bold'); axes[0].grid(True, alpha=0.5)
    names = list(split_models.keys()); r2s = [split_models[n]['r2'] for n in names]
    axes[1].bar(names, r2s, color=[ORNG, BLUE], alpha=0.85, edgecolor='white', width=0.5)
    for i, v in enumerate(r2s):
        axes[1].text(i, v+0.005, str(v), ha='center', fontsize=10, fontweight='bold')
    axes[1].set_ylim(0, 1.15); axes[1].set_ylabel('R² Score')
    axes[1].set_title('Temporal vs Random Split R²', fontsize=11, fontweight='bold'); axes[1].grid(True, axis='y', alpha=0.5)
    plt.tight_layout(); plots['leakage_analysis'] = fig_b64(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    feat_corr = corr[TARGET].drop(TARGET)
    colors_fc = [RED if abs(v)>0.9 else ORNG if abs(v)>0.5 else BLUE for v in feat_corr.values]
    ax.barh(feat_corr.index, feat_corr.values, color=colors_fc, alpha=0.85, edgecolor='white')
    ax.axvline(x=0.9, color=RED, linestyle='--', linewidth=1.5, label='High leak threshold (>0.9)')
    ax.axvline(x=-0.9, color=RED, linestyle='--', linewidth=1.5)
    ax.set_xlabel('Correlation with Target')
    ax.set_title('Feature-Target Correlation (Leakage Risk)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.5); plt.tight_layout()
    plots['feat_corr'] = fig_b64(fig)
    high_risk = [f for f in FEATURES if abs(corr.loc[f, TARGET]) > 0.9]
    return render_template('data_leakage.html', error=None, plots=plots, metrics={
        'n_samples': len(clean), 'features': FEATURES, 'target': TARGET,
        'corr_with_target': {f: round(corr.loc[f, TARGET], 4) for f in FEATURES},
        'high_risk_features': high_risk, 'split_comparison': split_models
    })
