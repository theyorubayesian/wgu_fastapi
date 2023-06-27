import os
import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42, max_depth=10, class_weight="balanced")
    model = model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def save_asset(asset, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(path, "wb") as f:
            asset = pickle.dump(asset, f)
    return asset

def load_asset(path):
    """ Loads pickle file from `path` and returns it."""
    with open(path, "rb") as f:
        asset = pickle.load(f)
    return asset

def evaluate_model(
    data, 
    cat_cols,
    output_dir,
    model=None,
    encoder=None,
    lb=None
):
    model = model or load_asset("trained_model.pkl")
    encoder = encoder or load_asset("encoder.pkl")
    lb = lb or load_asset("lb.pkl")

    performance_df = pd.DataFrame(columns=["feature", "precision", "recall", "fbeta"])
    for feature in cat_cols:
        feature_performance = []
        for category in data[feature].unique():
            mask = data[feature] == category
            subset = data[mask]
            X_test, y_test, *_ = process_data(
                subset,
                categorical_features=cat_cols,
                encoder=encoder,
                label="salary",
                training=False,
                lb=lb
            )
            y_pred = model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

            feature_performance.append(
                {
                    "feature": feature,
                    "category": category,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                }
            )

        performance_df = performance_df.append(feature_performance, ignore_index=True)
    output_file = os.path.join(output_dir, "slice_output.txt")
    performance_df.to_csv(output_file, index=False)
