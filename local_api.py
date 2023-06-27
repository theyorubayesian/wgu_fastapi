import json
import requests

url = "http://127.0.0.1:8000"

response = requests.get(url)
print("Status Code:", response.status_code)
print("Welcome Message:", response.text)


sample = {
 'age': 52,
 'workclass': 'Self-emp-not-inc',
 'fnlgt': 209642,
 'education': 'HS-grad',
 'education-num': 9,
 'marital-status': 'Married-civ-spouse',
 'occupation': 'Exec-managerial',
 'relationship': 'Husband',
 'race': 'White',
 'sex': 'Male',
 'capital-gain': 0,
 'capital-loss': 0,
 'hours-per-week': 45,
 'native-country': 'United-States',
}
url = "http://127.0.0.1:8000/model/"
headers = {"content-type": "application/json"} 
response = requests.post(url, data=json.dumps(sample), headers=headers)
print("Status Code:", response.status_code)
print("Prediction:", response.text)
