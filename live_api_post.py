import requests
import json

url = 'https://predict-salary.onrender.com/predict'

example = {
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

data = json.dumps(example)
resp = requests.post(url, data=data)

print(resp.status_code)
print(resp.json())
