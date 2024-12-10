import os
from flask import Flask, request, jsonify
import pandas as pd

from data_controller import DataController
from model_controller import ModelController


SCRIPT_DIR = os.path.dirname(__file__)
DATA_CONTROLLER_PATH = os.path.join(SCRIPT_DIR, 'checkpoints', 'data_controller.pkl')
MODEL_CONTROLLER_PATH = os.path.join(SCRIPT_DIR, 'checkpoints', 'model_controller.pkl')
TARGET_COLUMN = 'COMPANY_ID'

app = Flask(__name__)


def predictions_to_json_response(y_pred_proba, class_names):
    response = []
    for sample_probs in y_pred_proba:
        # Create a list of dictionaries with class name and probability
        sample_dict = [
            {"company_id": int(class_name), "probability": float(prob)}
            for class_name, prob in zip(class_names, sample_probs)
        ]
        # Sort the list by probability in descending order
        sorted_sample_dict = sorted(sample_dict, key=lambda x: x["probability"], reverse=True)
        response.append(sorted_sample_dict)
    return response

@app.route('/', methods=['GET'])
def index():
    return {}, 200

@app.route('/train', methods=['POST'])
def train():
    try:
        data_filename = request.json['data_filename']
        data_path = os.path.join(SCRIPT_DIR, 'dataset', data_filename)

        if not os.path.exists(data_path):
            return jsonify({'error': 'Data file not found'}), 404
        else:
            try:
                raw_dataframe = pd.read_parquet(data_path)
            except:
                return jsonify({'error': 'Could not read the data file'}), 500

        data_controller = DataController()
        dataset = data_controller.prepare_data(raw_dataframe=raw_dataframe, is_training=True)
        data_controller.save(filepath=DATA_CONTROLLER_PATH)

        model_controller = ModelController()
        model_controller.model_train(dataset=dataset)
        model_controller.save(filepath=MODEL_CONTROLLER_PATH)

        accuracy_top_1_train = round(model_controller.score_model_top_k_accuracy(k=1, dataset=dataset), 4)
        accuracy_top_20_train = round(model_controller.score_model_top_k_accuracy(k=20, dataset=dataset), 4)

        return jsonify({
            'message': 'Model trained successfully.',
            'accuracy_top_1_train': accuracy_top_1_train,
            'accuracy_top_20_train': accuracy_top_20_train
        }), 200
    except Exception as e:
            return jsonify({'error': 'Unexpected error happened. Check the input data.'}, 500)


@app.route('/score', methods=['POST'])
def score():
    try:
        data_filename = request.json['data_filename']
        data_path = os.path.join(SCRIPT_DIR, 'dataset', data_filename)

        if not os.path.exists(data_path):
            return jsonify({'error': 'Data file not found'}), 404
        else:
            try:
                raw_dataframe = pd.read_parquet(data_path)
            except:
                return jsonify({'error': 'Could not read the data file'}), 500
        
        if not os.path.exists(DATA_CONTROLLER_PATH) or not os.path.exists(MODEL_CONTROLLER_PATH):
            return jsonify({'error': 'Model not fitted yet. Please train the model first.'}), 400
        
        data_controller = DataController().load(filepath=DATA_CONTROLLER_PATH)
        model_controller = ModelController().load(filepath=MODEL_CONTROLLER_PATH)

        dataset = data_controller.prepare_data(raw_dataframe=raw_dataframe, is_training=False, force_include_target=True)

        accuracy_top_1 = round(model_controller.score_model_top_k_accuracy(k=1, dataset=dataset), 4)
        accuracy_top_20 = round(model_controller.score_model_top_k_accuracy(k=20, dataset=dataset), 4)
        
        return jsonify({
            'message': 'Successfully scored the model.',
            'accuracy_top_1': accuracy_top_1,
            'accuracy_top_20': accuracy_top_20
        }), 200
    except Exception as e:
            return jsonify({'error': 'Unexpected error happened. Check the input data or make sure the model is trained.'}, 500)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data = request.json['data']

        if not os.path.exists(DATA_CONTROLLER_PATH) or not os.path.exists(MODEL_CONTROLLER_PATH):
            return jsonify({'error': 'Model not fitted yet. Please train the model first.'}), 400

        data_controller = DataController().load(filepath=DATA_CONTROLLER_PATH)
        model_controller = ModelController().load(filepath=MODEL_CONTROLLER_PATH)
        
        raw_dataframe = pd.DataFrame(raw_data)
        data = data_controller.prepare_data(raw_dataframe=raw_dataframe, is_training=False)

        y_pred_proba = model_controller.model_predict(data=data)
        classes = model_controller.model.classes_
        predictions_json = predictions_to_json_response(y_pred_proba=y_pred_proba, class_names=classes)

        return jsonify({'message': 'Successfully inferenced the model.', 'predictions': predictions_json}), 200
    except Exception as e:
        return jsonify({'error': 'Unexpected error happened. Check the input data.'}, 500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
