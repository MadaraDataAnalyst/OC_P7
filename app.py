#uvicorn app:app --reload

from fastapi import FastAPI
import pickle
import pandas as pd
import uvicorn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create the app object
app = FastAPI()

# Load the model
with open('C:/Users/Madara/Documents/OC/OC_P7/pipeline_LGBM.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load client data
client_data = pd.read_csv('C:/Users/Madara/Documents/OC/OC_P7/df_V1_T.csv')

# Function returning all client IDs
@app.get("/client_list")
def load_client_id_list():
    id_list = client_data["SK_ID_CURR"].tolist()
    return id_list

# Function returning client data 
@app.get("/client_data")
def get_client_data(SK_ID_CURR: int):
    # Filter the client data for the specific SK_ID_CURR
    client_row = client_data[client_data['SK_ID_CURR'] == SK_ID_CURR]

    client_info = client_row.to_dict(orient="records")[0]

    # Convert any float values to strings
    for key, value in client_info.items():
        if isinstance(value, float):
            client_info[key] = str(value)

    return client_info

# Function returning, for a given client, the probability for defaulting
@app.get("/predict_default")
def predict_default(SK_ID_CURR: int):
    # Filter the client data for the specific SK_ID_CURR
    client_row = client_data[client_data['SK_ID_CURR'] == SK_ID_CURR]

    # Extract the features for prediction
    client_features = client_row.drop(columns=['TARGET'])

    # Make the prediction
    prediction = model.predict_proba(client_features)[:, 1].item()

    return {"SK_ID_CURR": SK_ID_CURR, "default_probability": prediction}

# Function returning, for a given client, the label for defaulting
@app.get("/predict_label")
def predict_label(SK_ID_CURR: int):
    # Filter the client data for the specific SK_ID_CURR
    client_row = client_data[client_data['SK_ID_CURR'] == SK_ID_CURR]

    # Extract the features for prediction
    client_features = client_row.drop(columns=['TARGET'])

    # Make the prediction
    prediction = model.predict(client_features).item()  # Use `model.predict` to get the label

    return {"SK_ID_CURR": SK_ID_CURR, "predict_label": prediction}
    
# Run the API with uvicorn. Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)