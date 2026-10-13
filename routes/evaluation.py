from flask import Blueprint, render_template
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import get_df, fig_b64, ORNG, BLUE, GRN, PURP, RED
from sklearn.model_selection import train_test_split, learning_curve, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

evaluation_bp = Blueprint('evaluation_bp', __name__)

_evaluation_cache = None

@evaluation_bp.route('/evaluation')
def evaluation_page():
    global _evaluation_cache
    if _evaluation_cache is not None:
        return _evaluation_cache

    df = get_df()
    if df is None:
        _evaluation_cache = render_template('evaluation.html', error='Dataset not found.')

    FEATURES = ['Year', 'Month', 'Unique Dispatched Vehicles', 'Total Dispatched Shared Trips']
    TARGET = 'Total Dispatched Trips'
    
    clean = df[FEATURES + [TARGET]].dropna()
    clean = clean[clean[TARGET] > 0]
    if len(clean) == 0:
        _evaluation_cache = render_template('evaluation.html', error='Not enough data.')
        
    clean = clean.sample(5000, random_state=42) if len(clean) > 10000 else clean
    X = clean[FEATURES].values
    y = clean[TARGET].values
    
    # For Simple Linear Regression
    X_simple = clean[['Unique Dispatched Vehicles']].values
    Xtr_simp, Xte_simp, ytr_simp, yte_simp = train_test_split(X_simple, y, test_size=0.2, random_state=42)
    
    # Train test split for all others
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data for linear models
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    
    models_info = {}
    
    # Helper function to evaluate
    def eval_model(name, model, yp, is_scaled=False, best_params=None):
        models_info[name] = {
            'model': model,
            'yp': yp,
            'r2': r2_score(yte, yp),
            'mse': mean_squared_error(yte, yp),
            'mae': mean_absolute_error(yte, yp),
            'rmse': np.sqrt(mean_squared_error(yte, yp)),
            'is_scaled': is_scaled
        }
        if best_params:
            models_info[name]['best_params'] = best_params

    # 1. Simple Linear Regression
    slr = LinearRegression().fit(Xtr_simp, ytr_simp)
    yp_slr = slr.predict(Xte_simp)
    eval_model('Simple Linear Regression', slr, yp_slr)

    # 2. Multiple Linear Regression (Baseline)
    lr = LinearRegression().fit(Xtr_s, ytr)
    yp_lr = lr.predict(Xte_s)
    eval_model('Multiple Linear Regression', lr, yp_lr, is_scaled=True)
    
    # 3. Ridge Regression
    ridge = Ridge(alpha=10.0).fit(Xtr_s, ytr)
    yp_ridge = ridge.predict(Xte_s)
    eval_model('Ridge Regression', ridge, yp_ridge, is_scaled=True)
    
    # 4. Decision Tree
    dt = DecisionTreeRegressor(max_depth=3, random_state=42).fit(Xtr, ytr)
    yp_dt = dt.predict(Xte)
    eval_model('Decision Tree', dt, yp_dt)
    
    # 5. AdaBoost
    ab = AdaBoostRegressor(n_estimators=50, random_state=42).fit(Xtr, ytr)
    yp_ab = ab.predict(Xte)
    eval_model('AdaBoost', ab, yp_ab)
    
    # 6. Hyperparameter Search (RandomizedSearchCV) for Gradient Boosting
    gb_param_dist = {
        'n_estimators': [20, 50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    gb_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), gb_param_dist, n_iter=3, cv=2, random_state=42, n_jobs=-1)
    gb_search.fit(Xtr, ytr)
    best_gb = gb_search.best_estimator_
    yp_gb = best_gb.predict(Xte)
    eval_model('Gradient Boosting (Random Search)', best_gb, yp_gb, best_params=gb_search.best_params_)
    
    # 7. Bayesian Optimization (BayesSearchCV) for Random Forest
    rf_opt = BayesSearchCV(
        RandomForestRegressor(random_state=42),
        {
            'n_estimators': Integer(20, 100),
            'max_depth': Integer(3, 10),
        },
        n_iter=3, cv=2, n_jobs=-1, random_state=42
    )
    rf_opt.fit(Xtr, ytr)
    best_rf = rf_opt.best_estimator_
    yp_rf = best_rf.predict(Xte)
    eval_model('Random Forest (Bayes Opt)', best_rf, yp_rf, best_params=rf_opt.best_params_)
    
    # Identify the best model based on R2
    best_model_name = max(models_info, key=lambda k: models_info[k]['r2'])
    best_model_info = models_info[best_model_name]
    
    # Statistical Testing (Paired t-test between Best Model and Baseline MLR on absolute errors)
    err_best = np.abs(yte - best_model_info['yp'])
    err_lr = np.abs(yte - models_info['Multiple Linear Regression']['yp'])
    t_stat, p_val = stats.ttest_rel(err_best, err_lr)
    stat_test = {
        'baseline': 'Multiple Linear Regression',
        'best': best_model_name,
        't_stat': round(t_stat, 4),
        'p_val': p_val,
        'significant': p_val < 0.05
    }
    
    # Diagnostic Curves for the Best Model
    plots = {}
    best_m = best_model_info['model']
    is_scaled = best_model_info['is_scaled']
    X_train_diag = Xtr_s if is_scaled else Xtr
    if best_model_name == 'Simple Linear Regression':
        X_train_diag = Xtr_simp
    
    # 1. Residual Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(best_model_info['yp'], yte - best_model_info['yp'], alpha=0.5, s=10, color=BLUE)
    ax.axhline(0, color=ORNG, linestyle='--', lw=2)
    ax.set_title(f'Residual Plot: {best_model_name}', fontweight='bold')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plots['residual'] = fig_b64(fig)
    
    # 2. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(yte, best_model_info['yp'], alpha=0.5, s=10, color=GRN)
    mn = min(yte.min(), best_model_info['yp'].min())
    mx = max(yte.max(), best_model_info['yp'].max())
    ax.plot([mn, mx], [mn, mx], color=RED, linestyle='--', lw=2)
    ax.set_title(f'Actual vs Predicted: {best_model_name}', fontweight='bold')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plots['actual_pred'] = fig_b64(fig)
    
    # 3. Learning Curve (Subset of data for speed)
    train_sizes, train_scores, test_scores = learning_curve(
        best_m, X_train_diag[:5000], ytr[:5000], cv=2, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='r2'
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, train_mean, 'o-', color=RED, label='Training score')
    ax.plot(train_sizes, test_mean, 'o-', color=GRN, label='Cross-validation score')
    ax.set_title('Learning Curve (Sampled Data)', fontweight='bold')
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('R2 Score')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plots['learning_curve'] = fig_b64(fig)
    
    # Formatting metrics for the template
    for name, info in models_info.items():
        info['r2'] = round(info['r2'], 4)
        info['mse'] = round(info['mse'], 2)
        info['mae'] = round(info['mae'], 2)
        info['rmse'] = round(info['rmse'], 2)

    _evaluation_cache = render_template('evaluation.html', error=None, models=models_info, best_name=best_model_name, stat_test=stat_test, plots=plots)

    return _evaluation_cache
