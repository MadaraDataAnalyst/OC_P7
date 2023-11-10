from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

# Create the Flask app object
app = Flask(__name__)

# Load the model
with open('C:/Users/Madara/Documents/OC/OC_P7/pipeline_LGBM.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load client data
client_data = pd.read_csv('C:/Users/Madara/Documents/OC/OC_P7/df_V1_T.csv')

# Function returning all client IDs
@app.route("/client_list", methods=["GET"])
def load_client_id_list():
    id_list = client_data["SK_ID_CURR"].tolist()
    return jsonify(id_list)

# Function returning client data
@app.route("/client_data", methods=["GET"])
def get_client_data():
    SK_ID_CURR = int(request.args.get("SK_ID_CURR"))

    # Filter the client data for the specific SK_ID_CURR
    client_row = client_data[client_data['SK_ID_CURR'] == SK_ID_CURR]

    client_info = client_row.to_dict(orient="records")[0]

    # Convert any float values to strings
    for key, value in client_info.items():
        if isinstance(value, float):
            client_info[key] = str(value)

    return jsonify(client_info)

# Function returning, for a given client, the probability for defaulting
@app.route("/predict_default", methods=["GET"])
def predict_default():
    SK_ID_CURR = int(request.args.get("SK_ID_CURR"))

    # Filter the client data for the specific SK_ID_CURR
    client_row = client_data[client_data['SK_ID_CURR'] == SK_ID_CURR]

    # Extract the features for prediction
    client_features = client_row.drop(columns=['TARGET'])

    # Make the prediction
    prediction = model.predict_proba(client_features)[:, 1].item()

    return jsonify({"SK_ID_CURR": SK_ID_CURR, "default_probability": prediction})

# Function returning, for a given client, the label for defaulting
@app.route("/predict_label", methods=["GET"])
def predict_label():
    SK_ID_CURR = int(request.args.get("SK_ID_CURR"))

    # Filter the client data for the specific SK_ID_CURR
    client_row = client_data[client_data['SK_ID_CURR'] == SK_ID_CURR]

    # Extract the features for prediction
    client_features = client_row.drop(columns=['TARGET'])

    # Make the prediction
    prediction = model.predict(client_features).item()  # Use `model.predict` to get the label

    return jsonify({"SK_ID_CURR": SK_ID_CURR, "predict_label": prediction})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
