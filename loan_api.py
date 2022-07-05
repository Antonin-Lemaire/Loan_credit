import json

import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
import pickle
from Kernel import get_data
from Modelisation_single import Instance, ClassifierAPI
import lime
import dill
from pydantic import BaseModel

app = FastAPI()

with open('api_model.pkl', 'rb') as pickle_in:
    clf = dill.load(pickle_in)
print('loading model complete')
df = pd.read_csv('test.csv')
print('loading dataset complete')


class ClientInput(BaseModel):
    id: int


class Prediction(BaseModel):
    pred: float


@app.get('/')
def index():
    return {'message': 'Loan default prediction model'}


# @app.get("/receive_index")
# async def receive_index():
#     return {'list_of_ids': df.index}


@app.post('/predict')
async def predict_outcome(data: ClientInput):
    print(data)
    unique_id = data.id
    feats = [f for f in df.columns if
             f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    df_input = df.loc[unique_id, feats]
    df_input.columns = feats
    print(df_input)

    prediction = clf.predict([df_input])
    pred = Prediction(pred=prediction)
    return pred


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
