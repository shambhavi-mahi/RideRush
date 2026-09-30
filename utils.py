import os, io, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "FHV_Base_Aggregate_Report_20260812.csv")

df_global = None

ORNG = '#F59E0B'; BLUE = '#3B82F6'; GRN = '#10B981'; PURP = '#8B5CF6'; RED = '#EF4444'

plt.rcParams.update({
    'figure.facecolor': '#f8fafc', 'axes.facecolor': '#ffffff',
    'axes.edgecolor':   '#0f172a', 'axes.labelcolor': '#475569',
    'xtick.color': '#475569',      'ytick.color': '#475569',
    'text.color':  '#0f172a',      'grid.color': '#0f172a', 'grid.alpha': 0.5,
})

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
