from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import os
#import pkg_resources
import logging
import logging.handlers

# Configure the root logger to print messages to the console
logging.basicConfig(level=logging.INFO)

#joblibversion = pkg_resources.get_distribution("joblib").version
#print(f"joblib version %r" % (joblibversion))

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Create the Flask app object
app = Flask(__name__)

# Create a separate logger for predictions
prediction_logger = logging.getLogger('prediction_logger')
prediction_logger.setLevel(logging.INFO)

# Create a file handler and set the log file
log_file = 'predictions.log'
file_handler = logging.FileHandler(log_file)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the prediction logger
prediction_logger.addHandler(file_handler)

# Load client data
client_data = pd.read_csv('C:/Users/Madara/Documents/OC/OC_P7/streamlit_cloud/df_V1_T_small.csv')          # local
#client_data = pd.read_csv('/home/MadaraRancane/mysite/df_V1_T_small.csv')                                 # web

def load_model():
    # Load the model
    model = load('C:/Users/Madara/Documents/OC/OC_P7/LGBMClassifier_with_nulls_model.joblib')                # local
    #model = load('/home/MadaraRancane/mysite/LGBMClassifier_with_nulls_model.joblib')                       # web
    return model

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

    model = load_model()

    # Make the prediction
    prediction = model.predict_proba(client_features)[:, 1].item()

    # Log the prediction
    prediction_logger.info(f"Prediction for SK_ID_CURR {SK_ID_CURR}: {prediction}")

    return jsonify({"SK_ID_CURR": SK_ID_CURR, "default_probability": prediction})

# Function returning, for a given client, the label for defaulting
@app.route("/predict_label", methods=["GET"])
def predict_label():
    SK_ID_CURR = int(request.args.get("SK_ID_CURR"))

    # Filter the client data for the specific SK_ID_CURR
    client_row = client_data[client_data['SK_ID_CURR'] == SK_ID_CURR]

    # Extract the features for prediction
    client_features = client_row.drop(columns=['TARGET'])

    model = load_model()

    # Make the prediction
    prediction = model.predict(client_features).item()  # Use `model.predict` to get the label

    return jsonify({"SK_ID_CURR": SK_ID_CURR, "predict_label": prediction})

# Run the Flask app
if __name__ == '__main__':
   app.run(host='127.0.0.1', port=8000, debug=True)                                       # local
   #app.run(debug=True)                                                                   # web
