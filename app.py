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

# ── LOGIN ──────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        # Demo: accept any non-empty credentials, redirect to dashboard
        if email and password:
            from flask import redirect
            return redirect('/load-data')
        error = 'Please enter both email and password.'
    return render_template('login.html', error=error)


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

# ── DBSCAN ─────────────────────────────────────────────────────────────────────
@app.route('/dbscan')
def dbscan_page():
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    df = get_df()
    if df is None:
        return render_template('dbscan.html', error='Dataset not found.', plots={}, metrics={})
    FEATURES = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips', 'Total Dispatched Trips']
    clean = df[FEATURES].dropna()
    clean = clean[clean['Total Dispatched Trips'] > 0].sample(n=min(len(clean), 2000), random_state=42)
    X = clean[FEATURES].values
    Xs = StandardScaler().fit_transform(X)
    X2 = PCA(n_components=2).fit_transform(Xs)
    plots = {}
    results = []
    for eps in [0.3, 0.5, 0.8, 1.2]:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(Xs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        try:
            sil = round(silhouette_score(Xs, labels), 4) if n_clusters > 1 else 'N/A'
        except Exception:
            sil = 'N/A'
        results.append({'eps': eps, 'clusters': n_clusters, 'noise': n_noise, 'silhouette': sil})
    db_best = DBSCAN(eps=0.5, min_samples=5)
    best_labels = db_best.fit_predict(Xs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(X2[:, 0], X2[:, 1], c=best_labels, cmap='tab10', alpha=0.7, s=15)
    axes[0].set_title('DBSCAN Clusters (eps=0.5) - PCA 2D', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('PCA 1'); axes[0].set_ylabel('PCA 2'); axes[0].grid(True, alpha=0.5)
    noise_mask = best_labels == -1
    axes[1].scatter(X2[~noise_mask, 0], X2[~noise_mask, 1], c=best_labels[~noise_mask], cmap='tab10', alpha=0.7, s=15, label='Cluster')
    axes[1].scatter(X2[noise_mask, 0], X2[noise_mask, 1], c='red', alpha=0.5, s=10, marker='x', label='Noise')
    axes[1].set_title('DBSCAN: Core vs Noise Points', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('PCA 1'); axes[1].set_ylabel('PCA 2')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.5)
    plt.tight_layout(); plots['dbscan_main'] = fig_b64(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    eps_vals = [r['eps'] for r in results]
    ax.plot(eps_vals, [r['clusters'] for r in results], 'o-', color=ORNG, linewidth=2, label='Clusters', markersize=8)
    ax2b = ax.twinx()
    ax2b.plot(eps_vals, [r['noise'] for r in results], 's--', color=RED, linewidth=2, label='Noise pts', markersize=8)
    ax.set_xlabel('eps value'); ax.set_ylabel('# Clusters', color=ORNG); ax2b.set_ylabel('# Noise Points', color=RED)
    ax.set_title('DBSCAN: eps Sensitivity Analysis', fontsize=11, fontweight='bold'); ax.grid(True, alpha=0.5)
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2b.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8)
    plt.tight_layout(); plots['eps_sensitivity'] = fig_b64(fig)
    return render_template('dbscan.html', error=None,
                           metrics={'n_samples': len(Xs), 'n_features': len(FEATURES), 'features': FEATURES, 'results': results},
                           plots=plots)

# ── PCA ────────────────────────────────────────────────────────────────────────
@app.route('/pca')
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

# ── ANOMALY DETECTION ──────────────────────────────────────────────────────────
@app.route('/anomaly')
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

# ── T-SNE / UMAP ───────────────────────────────────────────────────────────────
@app.route('/tsne-umap')
def tsne_umap_page():
    from sklearn.manifold import TSNE
    df = get_df()
    if df is None:
        return render_template('tsne_umap.html', error='Dataset not found.', plots={}, metrics={})
    FEATURES = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips', 'Total Dispatched Trips']
    clean = df[FEATURES].dropna()
    clean = clean[clean['Total Dispatched Trips'] > 0].sample(n=min(len(clean), 800), random_state=42)
    Xs = StandardScaler().fit_transform(clean[FEATURES].values)
    unique_years = sorted(clean['Year'].unique())
    year_map = {y: i for i, y in enumerate(unique_years)}
    c_arr = [year_map[y] for y in clean['Year'].values]
    plots = {}
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(Xs)
    pca2 = PCA(n_components=2); X_pca2 = pca2.fit_transform(Xs); ev = pca2.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=c_arr, cmap='tab10', alpha=0.7, s=15)
    axes[0].set_title('t-SNE 2D Projection (by Year)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(sc, ax=axes[0], label='Year index'); axes[0].grid(True, alpha=0.5)
    sc2 = axes[1].scatter(X_pca2[:, 0], X_pca2[:, 1], c=c_arr, cmap='tab10', alpha=0.7, s=15)
    axes[1].set_title('PCA 2D Projection (by Year)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({ev[0]*100:.1f}%)'); axes[1].set_ylabel(f'PC2 ({ev[1]*100:.1f}%)')
    plt.colorbar(sc2, ax=axes[1], label='Year index'); axes[1].grid(True, alpha=0.5)
    plt.tight_layout(); plots['tsne_pca'] = fig_b64(fig)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, perp in enumerate([10, 30, 50]):
        Xt = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=500).fit_transform(Xs)
        axes[i].scatter(Xt[:, 0], Xt[:, 1], c=c_arr, cmap='tab10', alpha=0.7, s=12)
        axes[i].set_title(f't-SNE perplexity={perp}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('t-SNE 1'); axes[i].set_ylabel('t-SNE 2'); axes[i].grid(True, alpha=0.5)
    plt.tight_layout(); plots['perplexity_comp'] = fig_b64(fig)
    return render_template('tsne_umap.html', error=None, plots=plots, metrics={
        'n_samples': len(clean), 'n_features': len(FEATURES), 'features': FEATURES,
        'unique_years': [int(y) for y in unique_years], 'perplexity': 30
    })

# ── DATA LEAKAGE ───────────────────────────────────────────────────────────────
@app.route('/data-leakage')
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
