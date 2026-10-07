from flask import Blueprint, render_template
from utils import get_df, fig_b64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

cluster_reg_bp = Blueprint('cluster_reg', __name__)

@cluster_reg_bp.route('/cluster-regression')
def cluster_regression_page():
    df = get_df()
    if df is None:
        return render_template('cluster_regression.html', error="Dataset not found.")

    try:
        # Prepare Data
        target = 'Total Dispatched Trips'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        if 'year' in numeric_cols: 
            numeric_cols.remove('year')
        
        # We need at least some features
        features = numeric_cols
        X = df[features].fillna(0)
        y = df[target].fillna(0)

        # Baseline Regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train_b, X_test_b, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        reg_base = LinearRegression()
        reg_base.fit(X_train_b, y_train)
        y_pred_b = reg_base.predict(X_test_b)
        
        base_metrics = {
            'r2': round(r2_score(y_test, y_pred_b), 4),
            'mae': round(mean_absolute_error(y_test, y_pred_b), 2),
            'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_b)), 2)
        }

        # Clustering for Augmentation
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Augment Features with Cluster Labels
        X_aug = X.copy()
        X_aug['Cluster'] = clusters
        
        # One-hot encode the cluster labels
        X_aug = pd.get_dummies(X_aug, columns=['Cluster'], drop_first=True)
        
        # Scale augmented features
        X_aug_scaled = StandardScaler().fit_transform(X_aug)
        
        # Augmented Regression
        X_train_a, X_test_a, _, _ = train_test_split(X_aug_scaled, y, test_size=0.2, random_state=42)
        
        reg_aug = LinearRegression()
        reg_aug.fit(X_train_a, y_train)
        y_pred_a = reg_aug.predict(X_test_a)
        
        aug_metrics = {
            'r2': round(r2_score(y_test, y_pred_a), 4),
            'mae': round(mean_absolute_error(y_test, y_pred_a), 2),
            'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_a)), 2)
        }

        # Calculate Improvement
        if base_metrics['mae'] > 0:
            improvement = round(((base_metrics['mae'] - aug_metrics['mae']) / base_metrics['mae']) * 100, 1)
        else:
            improvement = 0

        # Plotting comparisons
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: R2 Score Comparison
        axes[0].bar(['Baseline', 'Augmented'], [base_metrics['r2'], aug_metrics['r2']], color=['#94a3b8', '#3b82f6'])
        axes[0].set_title('R-Squared Score Comparison', fontsize=14)
        axes[0].set_ylabel('Score')
        axes[0].set_ylim(min(0, min(base_metrics['r2'], aug_metrics['r2']) - 0.1), 1.0)
        
        # Plot 2: Error Metrics (MAE & RMSE)
        x = np.arange(2)
        width = 0.35
        
        axes[1].bar(x - width/2, [base_metrics['mae'], aug_metrics['mae']], width, label='MAE', color='#f59e0b')
        axes[1].bar(x + width/2, [base_metrics['rmse'], aug_metrics['rmse']], width, label='RMSE', color='#ef4444')
        axes[1].set_title('Error Comparison (Lower is better)', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Baseline', 'Augmented'])
        axes[1].legend()

        plt.tight_layout()
        plot_b64 = fig_b64(fig)

        return render_template('cluster_regression.html', 
                               base=base_metrics, 
                               aug=aug_metrics, 
                               improvement=improvement, 
                               plot=plot_b64)

    except Exception as e:
        return render_template('cluster_regression.html', error=str(e))
