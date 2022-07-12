import flask
from flask import request, jsonify
import pandas as pd

app = flask.Flask(__name__)
app.config["DEBUG"] = True


feats = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_1',
         'EXT_SOURCE_2', 'EXT_SOURCE_3', 'NAME_INCOME_TYPE_Working', 'NAME_EDUCATION_TYPE_Higher education',
         'BURO_DAYS_CREDIT_MIN', 'BURO_DAYS_CREDIT_MEAN', 'BURO_DAYS_CREDIT_UPDATE_MEAN',
         'BURO_CREDIT_ACTIVE_Active_MEAN', 'BURO_CREDIT_ACTIVE_Closed_MEAN',
         'PREV_NAME_CONTRACT_STATUS_Approved_MEAN', 'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
         'PREV_CODE_REJECT_REASON_XAP_MEAN', 'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN',
         'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN', 'CC_CNT_DRAWINGS_CURRENT_MAX']
data = pd.read_csv('test.csv')
data = data[feats]
with open('api_model.pkl', 'rb') as pickle_in:
    clf = dill.load(pickle_in)


@app.route('/', methods=['GET'])
def home():
    return "<h1>Loan Credit Default Risk model</h1><p>This is an API designed to predict the defaulting risk of a " \
           "prospective client</p> "


@app.route('/prediction', methods=['GET'])
def api_prediction(request: Request):
    instance = pd.DataFrame(request['data'].to_dict())
    prediction_value = clf.predict(instance)
    prediction_context = clf.explain(instance)
    prediction_dict = {'value': prediction_value,
                       'context': prediction_context}
    return jsonify(prediction_dict)


if __name__ == '__main__':
    app.run()
