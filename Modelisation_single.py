import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from Kernel import get_data
from model import search
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from lime.lime_tabular import LimeTabularExplainer
import mlflow
from pydantic import BaseModel
import dill


class ClassifierAPI:
    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer

    def predict(self, data):
        output = self.model.predict(data)
        return output

    def explain(self, data: pd.DataFrame):
        context = self.explainer.explain_instance(data.values.T[0], self.model.predict, num_features=len(data.columns))
        return context


def train_explainer(data):
    feats = [f for f in data.columns if
             f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    feat_importance = LimeTabularExplainer(data[feats].values, mode='classification',
                                           feature_names=data[feats].columns)
    return feat_importance


def prediction(pipe, instance, explainer):
    """

    Parameters
    ----------
    pipe : mlflow model

    instance : 1-row dataframe, not containing target

    explainer : Lime model for visualisation of feature weights

    Returns
    -------
    output : expected value of target as per pipe prediction

    explanation : feature weights around instance
    """

    output = pipe.predict(instance)
    explanation = explainer.explain_instance(instance.values.T[0], pipe.predict, num_features=len(instance.columns))
    return output, explanation


def make_api(model, explainer, df_test):
    _ = 0
    return _


class Instance:
    def __init__(self, id, df):
        self.id = id
        self.columns = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
        self.values = df.loc[id, feats]


if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    feats = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    X = df[feats]
    y = df.TARGET
    explainer = train_explainer(df)

    with open('placeholder_mcdoctorate.sav', 'rb') as ph_dee:
        model = pickle.load(ph_dee)
    api_model = ClassifierAPI(model, explainer)
    dill.dump(api_model, open('api_model.pkl', 'wb'))




