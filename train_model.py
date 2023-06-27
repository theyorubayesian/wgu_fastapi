import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_asset,
    evaluate_model,
    save_asset,
    train_model,
)
PROJECT_DIR = "."
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_PATH = os.path.join(PROJECT_DIR, "data/clean_census.csv")



# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def main(cat_cols: list=cat_features, datapath: str=DATA_PATH):

    data = pd.read_csv(datapath)

    train, test = train_test_split(data, test_size=0.20)

    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, *_ = process_data(
        test, 
        categorical_features=cat_features, 
        label="salary", 
        training=False, 
        encoder=encoder, 
        lb=lb
    )

    # Train and save a model.
    model = train_model(X_train, y_train)
    assets_path = MODEL_DIR
    assets = [model, encoder, lb]
    assets_filenames = ["trained_model.pkl", "encoder.pkl", "lb.pkl"]

    for name, asset in zip(assets_filenames, assets):
        save_asset(asset, os.path.join(assets_path, name))

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    evaluate_model(data, cat_cols, OUTPUT_DIR, model, encoder, lb)
    return model, precision, recall, fbeta

if __name__ == "__main__":
    _, precision, recall, fbeta = main()
    print(f"Precision: {precision}")
    print(f"recall: {recall}")
    print(f"fbeta: {fbeta}")
