import os, logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from turnover_model import TurnoverModel
from domain import TurnoverInput
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


turnover_model_ = TurnoverModel.load_trained()


app = Flask(__name__)
CORS(app)

@app.route('/turnover/predict', methods=['POST'])
def predict_turnover():
    inp = TurnoverInput(request.json['nomenclature'], request.json['description'])
    output = turnover_model_.predict(inp)
    return jsonify(output)

app.run(host='0.0.0.0')
