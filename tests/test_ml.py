from unittest.mock import ANY
import ml.model as mod
import pytest_mock as mocker

def test_train_model(mocker, X_train, y_train):
    mock_model = mocker.patch("ml.model.RandomForestClassifier")

    _ = mod.train_model(X_train, y_train)

    mock_model.assert_called()
    mock_model().fit.assert_called_once_with(X_train, y_train)

def test_compute_model_metrics(mocker, y_test, preds):
    mock_precision_score = mocker.patch("ml.model.precision_score")
    mock_recall_score = mocker.patch("ml.model.recall_score")
    mock_fbeta_score = mocker.patch("ml.model.fbeta_score")

    _ = mod.compute_model_metrics(y_test, preds)

    mock_precision_score.assert_called_once_with(y_test, preds, zero_division=1)
    mock_recall_score.assert_called_once_with(y_test, preds, zero_division=1)
    mock_fbeta_score.assert_called_once_with(y_test, preds, beta=ANY, zero_division=1)
    


def test_inference(test_model, X_test, preds):
    inference = mod.inference(test_model, X_test)
    assert inference == preds