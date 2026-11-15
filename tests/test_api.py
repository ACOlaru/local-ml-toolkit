from fastapi.testclient import TestClient

from local_ml_toolkit.api import app, get_model


class FakeClassifier:
    def predict_many(self, texts):
        return [{"text": text, "label": "POSITIVE", "score": 0.99} for text in texts]


def test_predict_endpoint(monkeypatch):
    get_model.cache_clear()
    monkeypatch.setattr("local_ml_toolkit.api.get_model", lambda: FakeClassifier())
    client = TestClient(app)

    response = client.post("/predict", json={"texts": ["hello"]})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["label"] == "POSITIVE"
