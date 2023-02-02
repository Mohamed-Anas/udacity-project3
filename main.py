from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
import logging
from joblib import load
from starter.ml.data import process_data
from starter.ml.model import inference

model_location = './model/ml_model.joblib'
encoder_location = './model/encoder.joblib'
lb_location = './model/label_binarizer.joblib'

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

ml_model = load(model_location)
ml_encoder = load(encoder_location)
ml_lb = load(lb_location)
# Declare the data object with its components and their type.


class TaggedItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
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
        }


def get_salary(item):
    try:
        inp = item.dict(by_alias=True)
        temp_dict = {}
        for key, val in inp.items():
            temp_dict[key] = [val]
        df = pd.DataFrame.from_dict(temp_dict)
        logging.info(f'SUCCESS: Data read completed. Data shape {df.shape}')
        X_inf, _, _, _ = process_data(
            df, categorical_features=cat_features, label=None, training=False,
            encoder=ml_encoder, lb=ml_lb)
        logging.info(f'SUCESS: Data process complete {X_inf.shape}')
        y_inf = inference(ml_model, X_inf)
        logging.info(f'SUCESS: Inference completed {y_inf}')
        pred = ml_lb.inverse_transform(y_inf).ravel()[0]
        logging.info(f'SUCCESS: Output generated {pred}')
        return pred
    except Exception as e:
        logging.error(e)
        return 'Internal Error'


# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.


@app.get("/")
async def say_hello():
    return {"Hello User: Welcome to the Salary Prediction API"}


@app.post("/predict")
async def create_item(item: TaggedItem):
    salary = get_salary(item)
    return {"salary": salary}
