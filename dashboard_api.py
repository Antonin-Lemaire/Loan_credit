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
from flask import jsonify


def request_prediction(model_uri, value_id):
    headers = {"Content-Type": "application/json"}

    response = requests.post(
        model_uri+'predict', data=jsonify(value_id))
    response = json.loads(response.content.decode('utf-8'))
    return response


def request_index(model_uri):
    answer = requests.post(model_uri+'get_clients')

def modus_operandi():
    if not hasattr(modus_operandi, 'df'):
        modus_operandi.df = pd.read_csv('test.csv')
    print('loading data complete')
    df = modus_operandi.df
    API_URI = 'http://127.0.0.1:8000/'

    st.title('Loan credit default risk')

    choices = df.index
    client_id = st.selectbox('Select client ID',
                             choices)
    data = ClientInput(id=client_id)
    instance = df.loc[client_id, feats]
    instance.columns = feats
    predict_button = st.button('Predict')
    if predict_button:
        pred = request_prediction(API_URI, data)
        st.write(pred['value'])
        # st.write('This client\'s risk of defaulting on his loan is {:.2f}'.format(pred[0]))

        explanation = pred['context']
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)


if __name__ == '__main__':
    modus_operandi()
