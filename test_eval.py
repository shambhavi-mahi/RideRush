from app import app
from routes.evaluation import evaluation_page
with app.test_request_context('/evaluation'):
    print(evaluation_page()[:1000])
