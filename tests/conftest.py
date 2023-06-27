import pandas as pd
import pytest
from fastapi.testclient import TestClient
from main import app
from ml.model import load_asset


@pytest.fixture()
def X_train():
    return pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

@pytest.fixture()
def y_train():
    return pd.Series([10, 11, 12])

@pytest.fixture()
def X_test():
    return X_train

@pytest.fixture()
def y_test():
    return y_train

@pytest.fixture()
def preds():
    return y_test

@pytest.fixture()
def test_model(preds):
    class ModelMocker:
        def __init__(self, preds=preds):
            self.preds = preds

        def predict(self, X):
            return self.preds

    return ModelMocker()

@pytest.fixture()
def client():
    return TestClient(app)

@pytest.fixture()
def below_50k_sample():
    return {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
}


@pytest.fixture()
def above_50k_sample():
    return {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
}
