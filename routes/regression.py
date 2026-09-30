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


regression_bp = Blueprint('regression_bp', __name__)

@regression_bp.route('/regression')
def regression_page():
    df = get_df()
    if df is None:
        return render_template('regression.html', error='Dataset not found.', slr={}, mlr={}, plots={})
    TARGET = 'Total Dispatched Trips'
    SLRF   = 'Unique Dispatched Vehicles'
    MLRF   = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']
    plots  = {}

    clean = df[list(set([TARGET, SLRF] + MLRF))].dropna()
    clean = clean[clean[TARGET] > 0]
    y     = clean[TARGET].values

    # Simple Linear Regression
    Xs = clean[[SLRF]].values
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=.2, random_state=42)
    sm = LinearRegression().fit(Xtr, ytr); yp = sm.predict(Xte)
    slr = dict(feature=SLRF, target=TARGET,
               coef=round(float(sm.coef_[0]), 4), intercept=round(float(sm.intercept_), 4),
               r2=round(r2_score(yte,yp),4), mse=round(mean_squared_error(yte,yp),2),
               mae=round(mean_absolute_error(yte,yp),2),
               rmse=round(float(np.sqrt(mean_squared_error(yte,yp))),2),
               n_train=len(Xtr), n_test=len(Xte))
    slr['equation'] = f"Trips = {slr['coef']} * Vehicles + {slr['intercept']}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xl = np.linspace(Xs[:,0].min(), Xs[:,0].max(), 100).reshape(-1,1)
    axes[0].scatter(Xte.ravel(), yte, alpha=0.7, s=6, color=BLUE, label='Actual')
    axes[0].plot(xl, sm.predict(xl), color=ORNG, linewidth=2, label='Fit Line')
    axes[0].set_title('SLR: Vehicles -> Trips', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Vehicles'); axes[0].set_ylabel('Trips')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.7)
    axes[1].scatter(yp, yte-yp, alpha=0.7, s=6, color=GRN)
    axes[1].axhline(0, color=ORNG, linewidth=1.5, linestyle='--')
    axes[1].set_title('SLR Residuals', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residuals')
    axes[1].grid(True, alpha=0.7); plt.tight_layout()
    plots['slr'] = fig_b64(fig)

    # Multiple Linear Regression
    Xm = clean[MLRF].values
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xm, y, test_size=.2, random_state=42)
    ss = StandardScaler()
    Xtr2s = ss.fit_transform(Xtr2); Xte2s = ss.transform(Xte2)
    mm = LinearRegression().fit(Xtr2s, ytr2); yp2 = mm.predict(Xte2s)
    coefs = [dict(feature=f, coef=round(float(c), 4)) for f, c in zip(MLRF, mm.coef_)]
    mlr = dict(features=MLRF, target=TARGET, intercept=round(float(mm.intercept_), 4),
               r2=round(r2_score(yte2,yp2),4), mse=round(mean_squared_error(yte2,yp2),2),
               mae=round(mean_absolute_error(yte2,yp2),2),
               rmse=round(float(np.sqrt(mean_squared_error(yte2,yp2))),2),
               n_train=len(Xtr2), n_test=len(Xte2), coefs=coefs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mn = min(yte2.min(), yp2.min()); mx = max(yte2.max(), yp2.max())
    axes[0].scatter(yte2, yp2, alpha=0.7, s=6, color=BLUE)
    axes[0].plot([mn, mx], [mn, mx], color=ORNG, linewidth=2, linestyle='--', label='Perfect')
    axes[0].set_title('MLR: Actual vs Predicted', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.7)
    cv = [c['coef'] for c in coefs]
    fn = [c['feature'].replace('Total Dispatched ','').replace('Unique Dispatched ','') for c in coefs]
    axes[1].barh(fn, cv, color=[ORNG if v > 0 else RED for v in cv], edgecolor='#f8fafc', alpha=.85)
    axes[1].axvline(0, color='#0f172a', linewidth=.8, linestyle='--')
    axes[1].set_title('MLR Feature Coefficients (Scaled)', fontsize=11, fontweight='bold')
    axes[1].grid(True, axis='x', alpha=0.7); plt.tight_layout()
    plots['mlr'] = fig_b64(fig)

    # Logistic Regression
    y_bin = (y > np.median(y)).astype(int)
    Xtr3, Xte3, ytr3, yte3 = train_test_split(Xm, y_bin, test_size=.2, random_state=42)
    ss2 = StandardScaler()
    Xtr3s = ss2.fit_transform(Xtr3); Xte3s = ss2.transform(Xte3)
    lr = LogisticRegression().fit(Xtr3s, ytr3); yp3 = lr.predict(Xte3s)
    acc = accuracy_score(yte3, yp3)
    cm = confusion_matrix(yte3, yp3).tolist()
    logr = dict(features=MLRF, target="High Trips (> Median)",
                accuracy=round(acc, 4), n_train=len(Xtr3), n_test=len(Xte3), cm=cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Low', 'High']); ax.set_yticklabels(['Low', 'High'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Logistic Regression Confusion Matrix', fontsize=11, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', color='#0f172a' if cm[i][j] < np.max(cm)/2 else 'white', fontweight='bold')
    plots['logr'] = fig_b64(fig)

    # Ridge Regression (Regularization)
    alpha = 10.0
    ridge_model = Ridge(alpha=alpha).fit(Xtr2s, ytr2); yp_ridge = ridge_model.predict(Xte2s)
    ridge_coefs = [dict(feature=f, coef=round(float(c), 4)) for f, c in zip(MLRF, ridge_model.coef_)]
    ridge_reg = dict(features=MLRF, target=TARGET, intercept=round(float(ridge_model.intercept_), 4),
                     r2=round(r2_score(yte2, yp_ridge), 4), mse=round(mean_squared_error(yte2, yp_ridge), 2),
                     mae=round(mean_absolute_error(yte2, yp_ridge), 2),
                     rmse=round(float(np.sqrt(mean_squared_error(yte2, yp_ridge))), 2),
                     n_train=len(Xtr2), n_test=len(Xte2), coefs=ridge_coefs, alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(yte2, yp_ridge, alpha=0.7, s=6, color=BLUE)
    axes[0].plot([mn, mx], [mn, mx], color=ORNG, linewidth=2, linestyle='--', label='Perfect')
    axes[0].set_title('Ridge: Actual vs Predicted', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.7)
    cv_ridge = [c['coef'] for c in ridge_coefs]
    axes[1].barh(fn, cv_ridge, color=[ORNG if v > 0 else RED for v in cv_ridge], edgecolor='#f8fafc', alpha=.85)
    axes[1].axvline(0, color='#0f172a', linewidth=.8, linestyle='--')
    axes[1].set_title('Ridge Feature Coefficients (Scaled)', fontsize=11, fontweight='bold')
    axes[1].grid(True, axis='x', alpha=0.7); plt.tight_layout()
    plots['ridge'] = fig_b64(fig)

    return render_template('regression.html', error=None, slr=slr, mlr=mlr, logr=logr, ridge=ridge_reg, plots=plots)
