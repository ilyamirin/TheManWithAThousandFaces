import os
from flask import Flask, request, jsonify
from flask_cors import CORS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from core.turnover_model import TurnoverModel
from core.budget_model import BudgetModel
from core.domain import TurnoverInput, BudgetInput


turnover_model_ = TurnoverModel.load_trained()
budget_model_ = BudgetModel.load_trained()


app = Flask(__name__)
CORS(app)

@app.route('/turnover/predict', methods=['POST'])
def predict_turnover():
    inp = TurnoverInput(request.json['nomenclature'], request.json['description'])
    output = turnover_model_.predict(inp)
    return jsonify(output)

@app.route('/budget/predict', methods=['POST'])
def predict_budget():
    inp = BudgetInput(request.json['object'], request.json['project'], request.json['financing'])
    output = budget_model_.predict(inp)
    return jsonify(output)

app.run(host='0.0.0.0')
