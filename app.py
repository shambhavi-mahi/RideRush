import os, io, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix
import traceback

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "FHV_Base_Aggregate_Report_20260812.csv")

app = Flask(__name__, template_folder='templates', static_folder='static')

df_global      = None
model_pipeline = None

ORNG = '#F59E0B'; BLUE = '#3B82F6'; GRN = '#10B981'; PURP = '#8B5CF6'; RED = '#EF4444'

plt.rcParams.update({
    'figure.facecolor': '#f8fafc', 'axes.facecolor': '#ffffff',
    'axes.edgecolor':   '#0f172a', 'axes.labelcolor': '#475569',
    'xtick.color': '#475569',      'ytick.color': '#475569',
    'text.color':  '#0f172a',      'grid.color': '#0f172a', 'grid.alpha': 0.5,
})

# ── helpers ────────────────────────────────────────────────────────────────────
def parse_num(x):
    if pd.isna(x): return np.nan
    try:   return float(str(x).replace(',', '').strip())
    except: return np.nan

def get_df():
    global df_global
    if df_global is not None: return df_global
    if not os.path.exists(DATA_PATH): return None
    df = pd.read_csv(DATA_PATH)
    for col in ['Year', 'Month', 'Total Dispatched Trips',
                'Total Dispatched Shared Trips', 'Unique Dispatched Vehicles']:
        if col in df.columns:
            df[col] = df[col].apply(parse_num)
    df_global = df
    return df

def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150,
                facecolor='#f8fafc', edgecolor='none')
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── train GBM once at startup ──────────────────────────────────────────────────
def train_model():
    global model_pipeline
    df = get_df()
    if df is None: return
    FEATS  = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']
    TARGET = 'Total Dispatched Trips'
    clean  = df[FEATS + [TARGET]].dropna()
    clean  = clean[clean[TARGET] > 0]
    X, y   = clean[FEATS], clean[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('gb',  GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)),
    ])
    model_pipeline.fit(Xtr, ytr)
    print(f'[OK] GBM trained  R2={model_pipeline.score(Xte, yte):.4f}')

get_df()
train_model()

# ── static files (landing page) ────────────────────────────────────────────────
@app.route('/')
def index():      return send_from_directory(BASE_DIR, 'index.html')
@app.route('/style.css')
def css():        return send_from_directory(BASE_DIR, 'style.css', mimetype='text/css')
@app.route('/main.js')
def js():         return send_from_directory(BASE_DIR, 'main.js',  mimetype='application/javascript')
@app.route('/assets/<path:fn>')
def assets(fn):   return send_from_directory(os.path.join(BASE_DIR, 'assets'), fn)

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
@app.route('/load-data')
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

# ── EDA ────────────────────────────────────────────────────────────────────────
@app.route('/eda')
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

# ── GRAPHS ─────────────────────────────────────────────────────────────────────
@app.route('/graphs')
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

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────
@app.route('/feature-engg')
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

# ── REGRESSION ─────────────────────────────────────────────────────────────────
@app.route('/regression')
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

# ── TREES ──────────────────────────────────────────────────────────────────────
@app.route('/trees')
def trees_page():
    df = get_df()
    if df is None:
        return render_template('trees.html', error='Dataset not found.', tree={}, plots={})
    
    TARGET = 'Total Dispatched Trips'
    SLRF   = 'Unique Dispatched Vehicles'
    MLRF   = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']
    plots  = {}

    clean = df[list(set([TARGET, SLRF] + MLRF))].dropna()
    clean = clean[clean[TARGET] > 0]
    y     = clean[TARGET].values

    Xm = clean[MLRF].values
    Xtr, Xte, ytr, yte = train_test_split(Xm, y, test_size=.2, random_state=42)
    
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(Xtr, ytr)
    yp = dt.predict(Xte)

    tree_metrics = dict(
        features=MLRF, target=TARGET,
        r2=round(r2_score(yte, yp), 4),
        mse=round(mean_squared_error(yte, yp), 2),
        mae=round(mean_absolute_error(yte, yp), 2),
        rmse=round(float(np.sqrt(mean_squared_error(yte, yp))), 2),
        n_train=len(Xtr), n_test=len(Xte)
    )

    # Plot the tree
    fig, ax = plt.subplots(figsize=(20, 6))
    texts = plot_tree(dt, feature_names=MLRF, filled=True, ax=ax, fontsize=7, rounded=True)
    for text in texts:
        text.set_color('black')
    ax.set_title('Decision Tree Regressor (max_depth=3)', fontsize=12, fontweight='bold', color='#0f172a')
    plots['tree_plot'] = fig_b64(fig)

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    mn = min(yte.min(), yp.min()); mx = max(yte.max(), yp.max())
    ax.scatter(yte, yp, alpha=0.7, s=6, color=BLUE)
    ax.plot([mn, mx], [mn, mx], color=ORNG, linewidth=2, linestyle='--', label='Perfect')
    ax.set_title('Decision Tree: Actual vs Predicted', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.7)
    plots['scatter'] = fig_b64(fig)

    # Feature Importances
    fig, ax = plt.subplots(figsize=(7, 5))
    fi = dt.feature_importances_
    fn = [f.replace('Total Dispatched ','').replace('Unique Dispatched ','') for f in MLRF]
    ax.barh(fn, fi, color=ORNG, edgecolor='#f8fafc', alpha=.85)
    ax.set_title('Feature Importances', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.7); plt.tight_layout()
    plots['importances'] = fig_b64(fig)

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(Xtr, ytr)
    yp_rf = rf.predict(Xte)

    rf_metrics = dict(
        features=MLRF, target=TARGET,
        r2=round(r2_score(yte, yp_rf), 4),
        mse=round(mean_squared_error(yte, yp_rf), 2),
        mae=round(mean_absolute_error(yte, yp_rf), 2),
        rmse=round(float(np.sqrt(mean_squared_error(yte, yp_rf))), 2),
        n_train=len(Xtr), n_test=len(Xte)
    )

    # Plot actual vs predicted for RF
    fig, ax = plt.subplots(figsize=(7, 5))
    mn = min(yte.min(), yp_rf.min()); mx = max(yte.max(), yp_rf.max())
    ax.scatter(yte, yp_rf, alpha=0.7, s=6, color=BLUE)
    ax.plot([mn, mx], [mn, mx], color=ORNG, linewidth=2, linestyle='--', label='Perfect')
    ax.set_title('Random Forest: Actual vs Predicted', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.7)
    plots['rf_scatter'] = fig_b64(fig)

    # Feature Importances for RF
    fig, ax = plt.subplots(figsize=(7, 5))
    fi_rf = rf.feature_importances_
    ax.barh(fn, fi_rf, color=GRN, edgecolor='#f8fafc', alpha=.85)
    ax.set_title('RF Feature Importances', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.7); plt.tight_layout()
    plots['rf_importances'] = fig_b64(fig)

    # Gradient Boosting Regressor
    gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    gb.fit(Xtr, ytr)
    yp_gb = gb.predict(Xte)

    gb_metrics = dict(
        features=MLRF, target=TARGET,
        r2=round(r2_score(yte, yp_gb), 4),
        mse=round(mean_squared_error(yte, yp_gb), 2),
        mae=round(mean_absolute_error(yte, yp_gb), 2),
        rmse=round(float(np.sqrt(mean_squared_error(yte, yp_gb))), 2),
        n_train=len(Xtr), n_test=len(Xte)
    )

    # Plot actual vs predicted for GB
    fig, ax = plt.subplots(figsize=(7, 5))
    mn = min(yte.min(), yp_gb.min()); mx = max(yte.max(), yp_gb.max())
    ax.scatter(yte, yp_gb, alpha=0.7, s=6, color=BLUE)
    ax.plot([mn, mx], [mn, mx], color=ORNG, linewidth=2, linestyle='--', label='Perfect')
    ax.set_title('Gradient Boosting: Actual vs Predicted', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.7)
    plots['gb_scatter'] = fig_b64(fig)

    # Feature Importances for GB
    fig, ax = plt.subplots(figsize=(7, 5))
    fi_gb = gb.feature_importances_
    ax.barh(fn, fi_gb, color=PURP, edgecolor='#f8fafc', alpha=.85)
    ax.set_title('GB Feature Importances', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.7); plt.tight_layout()
    plots['gb_importances'] = fig_b64(fig)

    # AdaBoost Regressor
    ab = AdaBoostRegressor(n_estimators=50, random_state=42)
    ab.fit(Xtr, ytr)
    yp_ab = ab.predict(Xte)

    ab_metrics = dict(
        features=MLRF, target=TARGET,
        r2=round(r2_score(yte, yp_ab), 4),
        mse=round(mean_squared_error(yte, yp_ab), 2),
        mae=round(mean_absolute_error(yte, yp_ab), 2),
        rmse=round(float(np.sqrt(mean_squared_error(yte, yp_ab))), 2),
        n_train=len(Xtr), n_test=len(Xte)
    )

    # Plot actual vs predicted for AB
    fig, ax = plt.subplots(figsize=(7, 5))
    mn = min(yte.min(), yp_ab.min()); mx = max(yte.max(), yp_ab.max())
    ax.scatter(yte, yp_ab, alpha=0.7, s=6, color=BLUE)
    ax.plot([mn, mx], [mn, mx], color=ORNG, linewidth=2, linestyle='--', label='Perfect')
    ax.set_title('AdaBoost: Actual vs Predicted', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.7)
    plots['ab_scatter'] = fig_b64(fig)

    # Feature Importances for AB
    fig, ax = plt.subplots(figsize=(7, 5))
    fi_ab = ab.feature_importances_
    ax.barh(fn, fi_ab, color=RED, edgecolor='#f8fafc', alpha=.85)
    ax.set_title('AdaBoost Feature Importances', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.7); plt.tight_layout()
    plots['ab_importances'] = fig_b64(fig)

    return render_template('trees.html', error=None, tree=tree_metrics, rf=rf_metrics, gb=gb_metrics, ab=ab_metrics, plots=plots)

# ── CLUSTERING ─────────────────────────────────────────────────────────────────
@app.route('/clustering')
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

# ── PREDICT API ────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model not ready.'}), 500
    try:
        data  = request.get_json(force=True)
        FEATS = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']
        Xp    = pd.DataFrame([[float(data.get('year',2026)), float(data.get('month',1)),
                                float(data.get('vehicles',1)), float(data.get('shared',0))]], columns=FEATS)
        pred  = float(model_pipeline.predict(Xp)[0])
        bt    = str(data.get('baseType', 'medium'))
        final = max(10, round(pred * {'small':.85,'medium':1.0,'large':1.25,'enterprise':1.5}.get(bt, 1.0)))
        ep    = {'enterprise':.15, 'large':.18}.get(bt, .22)
        conf  = min(97, round(60 + np.log10(float(data.get('vehicles',1)) + 1) * 18))
        return jsonify({'predicted': final, 'low': round(final*(1-ep)), 'high': round(final*(1+ep)), 'confidence': conf})
    except Exception as e:
        traceback.print_exc(); return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print('\n  RideRush ML Server')
    print('  Landing   : http://127.0.0.1:5000')
    print('  Dashboard : http://127.0.0.1:5000/load-data\n')
    app.run(host='127.0.0.1', port=5000, debug=False)
