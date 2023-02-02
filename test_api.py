from fastapi.testclient import TestClient
import json
# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ['Hello User: Welcome to the Salary Prediction API']


def test_post_negative():
    example_neg = {
        'age': 28,
        'workclass': 'Private',
        'fnlgt': 338409,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Prof-specialty',
        'relationship': 'Wife',
        'race': 'Black',
                'sex': 'Female',
                'capital-gain': 0,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'Cuba'
    }
    data = json.dumps(example_neg)
    r = client.post("/predict", data=data)
    print(r.json())
    assert r.status_code == 200
    assert list(r.json().keys()) == ["salary"]
    assert r.json()["salary"] == " <=50K"


def test_post_positive():
    example_pos = {
        'age': 52,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': 209642,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'Wife',
                'sex': 'Male',
                'capital-gain': 14084,
                'capital-loss': 0,
                'hours-per-week': 45,
                'native-country': 'United-States'
    }
    data = json.dumps(example_pos)
    r = client.post("/predict", data=data)
    print(r.json())
    assert r.status_code == 200
    assert list(r.json().keys()) == ["salary"]
    assert r.json()["salary"] == " >50K"
