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


feature_engg_bp = Blueprint('feature_engg_bp', __name__)

@feature_engg_bp.route('/feature-engg')
def feature_engg_page():
    df = get_df()
    if df is None:
        return render_template('feature_engg.html', error='Dataset not found.',
                               numeric_cols=[], orig_rows=[], mm_rows=[], ss_rows=[], rs_rows=[],
                               nom_cols=[], nom_orig=[], nom_enc=[], encoders={}, ord_rows=[], ord_map={})
    NUM = [c for c in ['Year','Month','Total Dispatched Trips',
                        'Total Dispatched Shared Trips','Unique Dispatched Vehicles']
           if c in df.columns]
    nd = df[NUM].dropna().head(10).reset_index(drop=True)

    def scale(scaler):
        return pd.DataFrame(scaler.fit_transform(nd), columns=NUM).round(4).values.tolist()

    NOM = [c for c in ['Base Name', 'Month Name'] if c in df.columns]
    nom_orig, nom_enc, encoders = [], [], {}
    if NOM:
        ndf = df[NOM].head(10).fillna('Unknown')
        nom_orig = ndf.values.tolist()
        edf = ndf.copy()
        for col in NOM:
            le = LabelEncoder()
            edf[col] = le.fit_transform(ndf[col])
            encoders[col] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        nom_enc = edf.values.tolist()

    MO = ['January','February','March','April','May','June',
          'July','August','September','October','November','December']
    ord_map = {m: i+1 for i, m in enumerate(MO)}
    ord_rows = []
    if 'Month Name' in df.columns:
        od = df[['Month Name']].head(10).copy()
        od['Ordinal'] = od['Month Name'].map(ord_map).fillna(0).astype(int)
        ord_rows = od.fillna('--').values.tolist()

    return render_template('feature_engg.html', error=None,
        numeric_cols=NUM, orig_rows=nd.round(2).values.tolist(),
        mm_rows=scale(MinMaxScaler()), ss_rows=scale(StandardScaler()), rs_rows=scale(RobustScaler()),
        nom_cols=NOM, nom_orig=nom_orig, nom_enc=nom_enc, encoders=encoders,
        ord_rows=ord_rows, ord_map=ord_map)
