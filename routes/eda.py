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


eda_bp = Blueprint('eda_bp', __name__)

@eda_bp.route('/eda')
def eda_page():
    df = get_df()
    if df is None:
        return render_template('eda.html', error='Dataset not found.',
                               stats=[], missing=[], sample_cols=[], sample_rows=[],
                               total_rows=0, total_cols=0)
    NUM = [c for c in ['Year', 'Month', 'Total Dispatched Trips',
                        'Total Dispatched Shared Trips', 'Unique Dispatched Vehicles']
           if c in df.columns]
    desc  = df[NUM].describe(percentiles=[.25, .5, .75]).T
    stats = [dict(column=c, count=int(desc.loc[c, 'count']),
                  mean=round(desc.loc[c, 'mean'], 3), std=round(desc.loc[c, 'std'], 3),
                  min=round(desc.loc[c, 'min'], 3),  q1=round(desc.loc[c, '25%'], 3),
                  q2=round(desc.loc[c, '50%'], 3),   q3=round(desc.loc[c, '75%'], 3),
                  max=round(desc.loc[c, 'max'], 3)) for c in NUM]
    missing = [dict(column=c, dtype=str(df[c].dtype),
                    missing=int(df[c].isnull().sum()),
                    pct=round(df[c].isnull().sum() / len(df) * 100, 2))
               for c in df.columns]
    samp = df.sample(min(10, len(df)), random_state=99).fillna('--')
    return render_template('eda.html', error=None, stats=stats, missing=missing,
        sample_cols=df.columns.tolist(), sample_rows=samp.values.tolist(),
        total_rows=len(df), total_cols=len(df.columns))
