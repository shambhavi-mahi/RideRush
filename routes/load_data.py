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


load_data_bp = Blueprint('load_data_bp', __name__)

@load_data_bp.route('/load-data')
def load_data_page():
    df = get_df()
    if df is None:
        return render_template('load_data.html', error='Dataset not found: ' + DATA_PATH,
                               shape=None, dtypes={}, missing={}, miss_pct={},
                               columns=[], rows=[], total_rows=0)
    return render_template('load_data.html', error=None,
        shape=df.shape,
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        missing=df.isnull().sum().to_dict(),
        miss_pct=(df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        columns=df.columns.tolist(),
        rows=df.fillna('--').values.tolist(),
        total_rows=len(df))
