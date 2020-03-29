import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from core.turnover_model import TurnoverModel
from core.budget_model import BudgetModel
from core.domain import TurnoverInput, BudgetInput, BusinessException


turnover_model_ = TurnoverModel.load_trained()
budget_model_ = BudgetModel.load_trained()


app = Flask(__name__)
CORS(app)

@app.route('/turnover/predict', methods=['POST'])
def predict_turnover():
    inp = TurnoverInput(request.json['nomenclature'], request.json['description'])
    try:
        output = turnover_model_.predict(inp)
        return jsonify(output)
    except BusinessException as e:
        return jsonify(e.__dict__), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify(e.__dict__), 500

@app.route('/budget/predict', methods=['POST'])
def predict_budget():
    inp = BudgetInput(request.json['object'], request.json['project'], request.json['financing'])
    try:
        output = budget_model_.predict(inp)
        return jsonify(output)
    except BusinessException as e:
        return jsonify(e.__dict__), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify(e.__dict__), 500

app.run(host='0.0.0.0')
