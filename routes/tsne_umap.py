from flask import Blueprint, render_template
import numpy as np
import matplotlib.pyplot as plt
from utils import get_df, fig_b64, ORNG, BLUE, GRN, PURP, RED
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

tsne_umap_bp = Blueprint('tsne_umap_bp', __name__)

_tsne_cache = None

@tsne_umap_bp.route('/tsne-umap')
def tsne_umap_page():
    global _tsne_cache
    if _tsne_cache is not None:
        return _tsne_cache

    df = get_df()
    if df is None:
        _tsne_cache = render_template('tsne_umap.html', error='Dataset not found.', plots={}, metrics={})
        return _tsne_cache

    FEATURES = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips', 'Total Dispatched Trips']
    clean = df[FEATURES].dropna()
    clean = clean[clean['Total Dispatched Trips'] > 0]

    # ── Sample to max 500 rows — enough for visualisation, much faster ──
    clean = clean.sample(n=min(len(clean), 500), random_state=42)

    Xs = StandardScaler().fit_transform(clean[FEATURES].values)
    unique_years = sorted(clean['Year'].unique())
    year_map = {y: i for i, y in enumerate(unique_years)}
    c_arr = [year_map[y] for y in clean['Year'].values]

    plots = {}

    # ── t-SNE (single run, reduced iterations for speed) ────────
    X_tsne = TSNE(
        n_components=2, perplexity=30,
        random_state=42, max_iter=300,   # was 1000 → now 300
        n_jobs=-1
    ).fit_transform(Xs)

    # ── PCA 2D ──────────────────────────────────────────────────
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(Xs)
    ev = pca2.explained_variance_ratio_

    # Plot t-SNE + PCA side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=c_arr, cmap='tab10', alpha=0.7, s=15)
    axes[0].set_title('t-SNE 2D Projection (by Year)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(sc, ax=axes[0], label='Year index'); axes[0].grid(True, alpha=0.5)
    sc2 = axes[1].scatter(X_pca2[:, 0], X_pca2[:, 1], c=c_arr, cmap='tab10', alpha=0.7, s=15)
    axes[1].set_title('PCA 2D Projection (by Year)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({ev[0]*100:.1f}%)'); axes[1].set_ylabel(f'PC2 ({ev[1]*100:.1f}%)')
    plt.colorbar(sc2, ax=axes[1], label='Year index'); axes[1].grid(True, alpha=0.5)
    plt.tight_layout()
    plots['tsne_pca'] = fig_b64(fig)

    # ── Two perplexity variants only (removed 3rd run) ──────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, perp in enumerate([10, 50]):
        Xt = TSNE(
            n_components=2, perplexity=perp,
            random_state=42, max_iter=250, n_jobs=-1
        ).fit_transform(Xs)
        axes[i].scatter(Xt[:, 0], Xt[:, 1], c=c_arr, cmap='tab10', alpha=0.7, s=12)
        axes[i].set_title(f't-SNE perplexity={perp}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('t-SNE 1'); axes[i].set_ylabel('t-SNE 2')
        axes[i].grid(True, alpha=0.5)
    plt.tight_layout()
    plots['perplexity_comp'] = fig_b64(fig)

    _tsne_cache = render_template('tsne_umap.html', error=None, plots=plots, metrics={
        'n_samples': len(clean), 'n_features': len(FEATURES), 'features': FEATURES,
        'unique_years': [int(y) for y in unique_years], 'perplexity': 30
    })
    return _tsne_cache
