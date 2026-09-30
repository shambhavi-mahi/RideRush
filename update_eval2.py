import re

# --- Update templates/evaluation.html ---
html_content = """{% extends 'base.html' %}
{% set active = 'evaluation' %}

{% block content %}
<!-- Include MathJax for rendering formulas -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<div class="content-header">
  <div>
    <h1>Model Evaluation & Metrics</h1>
    <p>Formulas, Confusion Matrix, and Scikit-Learn implementations.</p>
  </div>
</div>

{% if error %}
<div class="alert alert-danger">{{ error }}</div>
{% else %}
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px;">

  <!-- REGRESSION METRICS -->
  <div class="card" style="padding: 24px;">
    <h2 style="font-size:18px; margin-bottom: 16px; display:flex; align-items:center; gap:8px;">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>
      Regression Metrics
    </h2>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;">
      <div style="background:#f8fafc; padding:16px; border-radius:12px;">
        <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase;">Mean Absolute Error (MAE)</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a; margin-top:4px;">{{ reg.mae }}</div>
        <div style="font-size:11px; margin-top:8px; color:#475569;">$$ \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| $$</div>
      </div>
      <div style="background:#f8fafc; padding:16px; border-radius:12px;">
        <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase;">Mean Squared Error (MSE)</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a; margin-top:4px;">{{ reg.mse }}</div>
        <div style="font-size:11px; margin-top:8px; color:#475569;">$$ \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 $$</div>
      </div>
      <div style="background:#f8fafc; padding:16px; border-radius:12px;">
        <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase;">Root Mean Squared (RMSE)</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a; margin-top:4px;">{{ reg.rmse }}</div>
        <div style="font-size:11px; margin-top:8px; color:#475569;">$$ \text{RMSE} = \sqrt{\text{MSE}} $$</div>
      </div>
      <div style="background:#f8fafc; padding:16px; border-radius:12px;">
        <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase;">R-Squared (R&sup2;)</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a; margin-top:4px;">{{ reg.r2 }}</div>
        <div style="font-size:11px; margin-top:8px; color:#475569;">$$ R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} $$</div>
      </div>
    </div>
    
    <div style="font-size:14px; font-weight:600; margin-bottom:8px;">Scikit-Learn Implementation</div>
    <pre style="background:#0f172a; color:#f8fafc; padding:16px; border-radius:12px; font-size:13px; overflow-x:auto;"><code>from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)</code></pre>
  </div>

  <!-- CLASSIFICATION METRICS -->
  <div class="card" style="padding: 24px;">
    <h2 style="font-size:18px; margin-bottom: 16px; display:flex; align-items:center; gap:8px;">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
      Classification Metrics
    </h2>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;">
      <div style="background:#f8fafc; padding:16px; border-radius:12px;">
        <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase;">Accuracy</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a; margin-top:4px;">{{ clf.accuracy }}</div>
        <div style="font-size:11px; margin-top:8px; color:#475569;">$$ \frac{TP + TN}{TP + TN + FP + FN} $$</div>
      </div>
      <div style="background:#f8fafc; padding:16px; border-radius:12px;">
        <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase;">Precision</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a; margin-top:4px;">{{ clf.precision }}</div>
        <div style="font-size:11px; margin-top:8px; color:#475569;">$$ \frac{TP}{TP + FP} $$</div>
      </div>
      <div style="background:#f8fafc; padding:16px; border-radius:12px;">
        <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase;">Recall</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a; margin-top:4px;">{{ clf.recall }}</div>
        <div style="font-size:11px; margin-top:8px; color:#475569;">$$ \frac{TP}{TP + FN} $$</div>
      </div>
      <div style="background:#f8fafc; padding:16px; border-radius:12px;">
        <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase;">F1 Score</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a; margin-top:4px;">{{ clf.f1 }}</div>
        <div style="font-size:11px; margin-top:8px; color:#475569;">$$ 2 \times \frac{\text{Prec} \times \text{Rec}}{\text{Prec} + \text{Rec}} $$</div>
      </div>
    </div>

    <!-- CONFUSION MATRIX -->
    <div style="margin-bottom: 24px;">
      <div style="font-size:14px; font-weight:600; margin-bottom:12px;">Confusion Matrix</div>
      <table style="width:100%; border-collapse:collapse; background:#f8fafc; border-radius:12px; overflow:hidden; font-size:14px;">
        <tr>
          <td style="border:1px solid #e2e8f0; padding:12px; text-align:center;">
            <div style="font-size:11px; color:#64748b; font-weight:600; text-transform:uppercase; margin-bottom:4px;">True Negatives</div>
            <div style="font-size:20px; font-weight:700; color:#0f172a;">{{ clf.cm[0][0] }}</div>
          </td>
          <td style="border:1px solid #e2e8f0; padding:12px; text-align:center; background:rgba(239,68,68,0.06);">
            <div style="font-size:11px; color:#ef4444; font-weight:600; text-transform:uppercase; margin-bottom:4px;">False Positives</div>
            <div style="font-size:20px; font-weight:700; color:#b91c1c;">{{ clf.cm[0][1] }}</div>
          </td>
        </tr>
        <tr>
          <td style="border:1px solid #e2e8f0; padding:12px; text-align:center; background:rgba(239,68,68,0.06);">
            <div style="font-size:11px; color:#ef4444; font-weight:600; text-transform:uppercase; margin-bottom:4px;">False Negatives</div>
            <div style="font-size:20px; font-weight:700; color:#b91c1c;">{{ clf.cm[1][0] }}</div>
          </td>
          <td style="border:1px solid #e2e8f0; padding:12px; text-align:center; background:rgba(16,185,129,0.1);">
            <div style="font-size:11px; color:#10b981; font-weight:600; text-transform:uppercase; margin-bottom:4px;">True Positives</div>
            <div style="font-size:20px; font-weight:700; color:#047857;">{{ clf.cm[1][1] }}</div>
          </td>
        </tr>
      </table>
    </div>

    <div style="font-size:14px; font-weight:600; margin-bottom:8px;">Scikit-Learn Implementation</div>
    <pre style="background:#0f172a; color:#f8fafc; padding:16px; border-radius:12px; font-size:13px; overflow-x:auto;"><code>from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)</code></pre>
  </div>

</div>
{% endif %}
{% endblock %}
"""

with open('templates/evaluation.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print('Updated evaluation page with formulas and confusion matrix.')
