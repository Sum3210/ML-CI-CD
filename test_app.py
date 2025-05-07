import json
from app import app

def test_home():
    response = app.test_client().get('/')
    assert response.status_code == 200
    assert "Witaj" in response.get_json()["message"]

def test_info():
    response = app.test_client().get('/info')
    data = response.get_json()
    assert response.status_code == 200
    assert data["model_type"] == "LogisticRegression"
    assert "setosa" in data["target_classes"]

def test_health():
    response = app.test_client().get('/health')
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"

def test_predict_valid():
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = app.test_client().post('/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "class_name" in data

def test_predict_missing_data():
    response = app.test_client().post('/predict', data=json.dumps({}), content_type='application/json')
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_predict_invalid_shape():
    payload = {"features": [1.0, 2.0]}  # Za mało danych
    response = app.test_client().post('/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 400
    assert "error" in response.get_json()
