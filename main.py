import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_asset
from fastapi.responses import JSONResponse


app = FastAPI()


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


encoder = load_asset("./model/encoder.pkl")
lb = load_asset("./model/lb.pkl")
model = load_asset("./model/trained_model.pkl")

@app.on_event("startup")
async def startup_event(): 
    global model, encoder, lb
    model = load_asset("./model/trained_model.pkl")
    encoder = load_asset("./model/encoder.pkl")
    lb = load_asset("./model/lb.pkl")




@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={"Message": "Welcome to Igbanladogi App! It may seem like you are interested in making some predictions..."},
    )

@app.post("/model/")
async def predict(data: Data):
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

    df = pd.DataFrame(data.dict(by_alias=True), index=[0])

    X, *_ = process_data(
        df, 
        categorical_features=cat_features, 
        label=None , 
        encoder=encoder,
        training=False, 
        lb=lb)

    predictions = inference(model, X)
    predictions = apply_label(predictions)
    return JSONResponse(status_code=200, content=predictions)
