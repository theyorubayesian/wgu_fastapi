from fastapi.testclient import TestClient


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "Message": "Welcome to Igbanladogi App! It may seem like you are interested in making some predictions..."
    }

def test_predict_below_50k(client, below_50k_sample):
    r = client.post("/model/", json=below_50k_sample)
    assert r.status_code == 200
    assert r.json() == "<=50K"

def test_predict_above_50k(client, above_50k_sample):
    r = client.post("/model/", json=above_50k_sample)
    assert r.status_code == 200
    assert r.json() == ">50K"
