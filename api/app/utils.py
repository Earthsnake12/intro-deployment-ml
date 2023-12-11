from joblib import load
from scipy.sparse import data
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame
import os 
from io import BytesIO

import pickle


def get_model() -> Pipeline:
    model_path = os.environ.get('MODEL_PATH','model/model.pkl')
    with open(model_path, "rb") as fh:
        model = load(BytesIO(fh.read()))
    print(model)
    return model

def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    print(class_model)
    transition_dictionary = {key:[value] for key, value in class_model.dict().items()}
    data_frame = DataFrame(transition_dictionary)
    print(data_frame)
    return data_frame
