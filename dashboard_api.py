import streamlit as st
import pandas as pd
import numpy as np
from Kernel import get_data
import dill
import json
import requests
from Modelisation_single import ClassifierAPI
import matplotlib.pyplot as plt
from loan_api import ClientInput

with open('api_model.pkl', 'rb') as pickle_in:
    clf = dill.load(pickle_in)


def request_prediction(model_uri, value_id):
    headers = {"Content-Type": "application/json"}

    response = requests.post(
        model_uri+'predict', data=value_id.json())

    return response


def modus_operandi():
    if not hasattr(modus_operandi, 'df'):
        modus_operandi.df = pd.read_csv('test.csv')
    print('loading data complete')
    df = modus_operandi.df
    API_URI = 'http://127.0.0.1:8000/'

    st.title('Loan credit default risk')
    feats = [f for f in df.columns if
             f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    choices = df.index
    client_id = st.selectbox('Select client ID',
                             choices)
    data = ClientInput(id=client_id)
    instance = df.loc[client_id, feats]
    instance.columns = feats
    predict_button = st.button('Predict')
    if predict_button:
        pred = request_prediction(API_URI, data)
        pred = pred.json()
        st.write(pred['pred'])
        # st.write('This client\'s risk of defaulting on his loan is {:.2f}'.format(pred[0]))

        explanation = clf.explain(instance)
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)


if __name__ == '__main__':
    modus_operandi()
