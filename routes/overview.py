from flask import Blueprint, render_template
from utils import get_df

overview_bp = Blueprint('overview_bp', __name__)

@overview_bp.route('/overview')
@overview_bp.route('/dashboard')
def overview_page():
    df = get_df()
    total_rows = 0
    if df is not None:
        total_rows = len(df)
    return render_template('overview.html', total_rows=total_rows)
