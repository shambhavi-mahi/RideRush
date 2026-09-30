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


graphs_bp = Blueprint('graphs_bp', __name__)

@graphs_bp.route('/graphs')
def graphs_page():
    df = get_df()
    if df is None:
        return render_template('graphs.html', error='Dataset not found.', plots={})
    plots = {}

    def hist_plot(col, title, color):
        fig, ax = plt.subplots(figsize=(7, 4))
        d = df[col].dropna(); d = d[d <= d.quantile(.99)]
        ax.hist(d, bins=55, color=color, alpha=.8, edgecolor='#f8fafc')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.7)
        return fig_b64(fig)

    plots['hist_trips'] = hist_plot('Total Dispatched Trips', 'Distribution — Total Dispatched Trips', ORNG)
    plots['hist_veh']   = hist_plot('Unique Dispatched Vehicles', 'Distribution — Unique Dispatched Vehicles', BLUE)

    # Trips by Month
    fig, ax = plt.subplots(figsize=(9, 4))
    mo = df.groupby('Month')['Total Dispatched Trips'].mean().sort_index()
    bars = ax.bar(mo.index, mo.values, color=ORNG, edgecolor='#f8fafc', width=.6)
    ax.set_title('Avg Trips by Month', fontsize=11, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=8)
    ax.grid(True, axis='y', alpha=0.7)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{b.get_height():.0f}',
                ha='center', va='bottom', fontsize=7, color='#475569')
    plots['bar_month'] = fig_b64(fig)

    # Trips by Year
    fig, ax = plt.subplots(figsize=(9, 4))
    yr = df.groupby('Year')['Total Dispatched Trips'].mean().sort_index()
    ax.plot(yr.index, yr.values, color=ORNG, marker='o', linewidth=2)
    ax.fill_between(yr.index, yr.values, alpha=.15, color=ORNG)
    ax.set_title('Avg Trips by Year', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.7)
    plots['line_year'] = fig_b64(fig)

    # Scatter: Vehicles vs Trips
    fig, ax = plt.subplots(figsize=(7, 5))
    s = df[['Unique Dispatched Vehicles','Total Dispatched Trips']].dropna().sample(min(3000,len(df)),random_state=1)
    ax.scatter(s['Unique Dispatched Vehicles'], s['Total Dispatched Trips'], alpha=0.7, s=6, color=BLUE)
    ax.set_title('Vehicles vs Total Dispatched Trips', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.7)
    plots['scatter_vt'] = fig_b64(fig)

    # Scatter: Shared vs Trips
    fig, ax = plt.subplots(figsize=(7, 5))
    s2 = df[['Total Dispatched Shared Trips','Total Dispatched Trips']].dropna()
    s2 = s2[s2['Total Dispatched Shared Trips'] > 0]
    if len(s2) > 0:
        s2 = s2.sample(min(3000, len(s2)), random_state=2)
        ax.scatter(s2['Total Dispatched Shared Trips'], s2['Total Dispatched Trips'], alpha=0.7, s=6, color=GRN)
    ax.set_title('Shared Trips vs Total Trips', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.7)
    plots['scatter_st'] = fig_b64(fig)

    # Correlation heatmap
    NUM = [c for c in ['Year','Month','Total Dispatched Trips','Total Dispatched Shared Trips','Unique Dispatched Vehicles'] if c in df.columns]
    corr = df[NUM].dropna().corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    lbls = ['Year', 'Month', 'Trips', 'Shared', 'Veh']
    ax.set_xticks(range(len(lbls))); ax.set_yticks(range(len(lbls)))
    ax.set_xticklabels(lbls, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(lbls, fontsize=9)
    for i in range(len(lbls)):
        for j in range(len(lbls)):
            ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=9, color='#0f172a', fontweight='bold')
    ax.set_title('Correlation Heatmap', fontsize=11, fontweight='bold')
    plots['heatmap'] = fig_b64(fig)

    # Box plots
    BCOLS = [c for c in ['Total Dispatched Trips','Unique Dispatched Vehicles','Total Dispatched Shared Trips'] if c in df.columns]
    fig, axes = plt.subplots(1, len(BCOLS), figsize=(5*len(BCOLS), 5))
    if len(BCOLS) == 1: axes = [axes]
    for ax, col, clr in zip(axes, BCOLS, [ORNG, BLUE, GRN]):
        d = df[col].dropna(); d = d[d <= d.quantile(.99)]
        ax.boxplot(d, patch_artist=True,
                   boxprops=dict(facecolor=clr, alpha=.7),
                   medianprops=dict(color='#0f172a', linewidth=2),
                   whiskerprops=dict(color='#475569'), capprops=dict(color='#475569'),
                   flierprops=dict(marker='.', color='#475569', markersize=3, alpha=0.7))
        ax.set_title(col.replace('Total Dispatched ','').replace('Unique Dispatched ',''), fontsize=10)
        ax.grid(True, axis='y', alpha=0.7)
    fig.suptitle('Box Plots', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plots['boxplot'] = fig_b64(fig)

    # Top 10 bases
    if 'Base Name' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        top = df.groupby('Base Name')['Total Dispatched Trips'].sum().nlargest(10)
        ax.barh(list(range(len(top))), top.values, color=PURP, edgecolor='#f8fafc', alpha=.85)
        ax.set_yticks(list(range(len(top))))
        ax.set_yticklabels([b[:35] for b in top.index], fontsize=8)
        ax.set_title('Top 10 Bases by Total Trips', fontsize=11, fontweight='bold')
        ax.set_xlabel('Total Trips'); ax.grid(True, axis='x', alpha=0.7)
        plots['top_bases'] = fig_b64(fig)

    return render_template('graphs.html', plots=plots, error=None)
