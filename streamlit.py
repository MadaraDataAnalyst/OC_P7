import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from PIL import Image
import shap
import requests

# Set the page layout to wide
st.set_page_config(layout="wide")

# Title and subtitle
st.title("Prêt à dépenser - Default risk calculator Dashboard")
st.write(
    "Prêt à dépenser's app assesses loan applicants' creditworthiness and predicts the likelihood (in %) of loan default. It also allows you to view personal data of a selected client and compare it with others for transparent credit decisions."
)

# API URL for Flask
#API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = 'https://madararancane.eu.pythonanywhere.com'

# Function to fetch data from the Flask API
def fetch_data_from_api(endpoint, params=None):
    response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params)
    return response.json()

# Load trained model
#LGBM_model = joblib.load("C:/Users/Madara/Documents/OC/OC_P7/LGBMClassifier_with_nulls_model.joblib")  #local
LGBM_model = joblib.load('LGBMClassifier_with_nulls_model.joblib')           

# Load the DataFrame containing the client data (df_V1_T)
#df = pd.read_csv("C:/Users/Madara/Documents/OC/OC_P7/df_V1_T.csv")                                     #local
df = pd.read_csv('streamlit_cloud/df_V1_T_small.csv')                                        
df = df.drop(["TARGET"], axis=1)

# Load SHAP values
#shap_values = joblib.load("C:/Users/Madara/Documents/OC/OC_P7/shap_values.pkl")                        #local
shap_values = joblib.load('shap_values.pkl')                                 

############## FUNCTIONS

def plot_risk(proba, threshold=50, max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Default Probability (%) for a Credit Applicant", 'font': {'size': 26, 'color': "rgb(51, 51, 51)"}},
        delta={'reference': threshold, 'increasing': {'color': "#d62728"}, 'decreasing': {'color': "#2ca02c"}},
        gauge={'axis': {'range': [0, max_val], 'dtick': 10, 'tickwidth': 1, 'tickcolor': "rgb(51, 51, 51)"},
               'bar': {'color': "white"},
               'steps': [{'range': [0, 45], 'color': '#2ca02c'},
                         {'range': [45, 65], 'color': '#f7eb7c'},
                         {'range': [65, max_val], 'color': '#d62728'}],
               'threshold': {
                   'line': {'color': 'rgb(51, 51, 51)', 'width': 5},
                   'thickness': 0.8, 
                   'value': proba,

               }
        }
    ))

    # Update layout for better appearance and accessibility
    fig.update_layout(paper_bgcolor="white", font={"color": "rgb(51, 51, 51)"})

    return fig

############## SIDEBAR

# Insert logo
image = Image.open("LOGO.png")
st.sidebar.image(image)

# Sidebar to input user's client ID
st.sidebar.title("Client Selection")

# Fetch client IDs from the Flask API
client_ids = fetch_data_from_api("client_list")

# Add "Client ID" add as the default option
client_ids.insert(0, "Client ID")

# Select a client ID
client_id = st.sidebar.selectbox("Select a Client ID", client_ids, index=0)

# Check if the selected client ID is "Client ID"
if client_id == "Client ID":
    st.warning("Please pick a client ID from the panel on the left")
else:
    # Fetch the default probability prediction from the Flask API
    prediction_data = fetch_data_from_api(
        "predict_default", params={"SK_ID_CURR": client_id}
    )
    client_score = prediction_data.get("default_probability")

    client_score_percentage = client_score * 100
    # Create and display the gauge chart for client_score_percentage
    fig = plot_risk(client_score_percentage)
    st.plotly_chart(fig)

    # Fetch the label prediction from the `/predict_label` endpoint
    prediction_label = fetch_data_from_api(
        "predict_label", params={"SK_ID_CURR": client_id}
    )
    client_label = prediction_label.get("predict_label")

    # Create an expander for Feature Impact on Default Risk
    with st.expander("Explain the decision"):
        # Interpretation using SHAP values
        client_id = int(client_id)
        client_instance = df[df['SK_ID_CURR'] == client_id]

        # Determine which SHAP values to use based on the label
        if client_label == 0:
            shap_values_instance = shap_values[0][client_instance.index[0], :]
            title = "These features led to approval"
        else:
            shap_values_instance = shap_values[1][client_instance.index[0], :]
            title = "These features led to rejection"

        top_indices = np.argsort(-shap_values_instance)  # Sort in descending order
        feature_names = df.columns

        top_features = [feature_names[i] for i in top_indices[:5]]
        contributions = shap_values_instance[top_indices[:5]]

        # Display the information and lists as comma-separated strings
        st.write(f"**{title}**: {', '.join(top_features)}")

        # Get the expected value for the positive class by making predictions
        expected_value_positive = -0.544

        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)

        # Custom colors
        positive_color = "#ff7f0e"
        negative_color = "#1f77b4"

        # Plot shap values
        st.markdown(
            "<h2 style='text-align: center; font-size: 16px;'>Feature Impact on Default Risk</h2>",
            unsafe_allow_html=True)
        st_shap(shap.force_plot(expected_value_positive, shap_values_instance, client_instance, plot_cmap=[positive_color, negative_color]))

    # Fetch client personal data
    client_data = fetch_data_from_api("client_data", params={"SK_ID_CURR": client_id})

    # Create an expander to view client personal data
    with st.expander("View client personal data"):
        # Create a multiselect to select multiple features to display
        selected_features = st.multiselect("Select Features", list(client_data.keys()))

        # Display the selected features and their values for the specific client
        for feature in selected_features:
            if feature in client_data:
                value = client_data[feature]
                st.write(f"**{feature}**: {value}")

    # Create a new expander for Histogram with a single feature dropdown
    with st.expander("Compare client to others"):

        # Predict the class for every row in the dataset and add it as a new column 'PREDICTED_CLASS'
        df['PREDICTED_CLASS'] = LGBM_model.predict(df)

        # Get the selected client's information
        client_id = int(client_id)
        client_info = df[df['SK_ID_CURR'] == client_id]

        # Create a dropdown to select a single feature for comparison
        selected_feature_comp = st.selectbox("Select a feature for comparison", list(client_info.columns))

        col1, col2 = st.columns(2)
        with col1:

            # Create a Seaborn histogram plot with 'hue' parameter
            plt.figure(figsize=(8, 6))
            if selected_feature_comp in client_info:
                # Filter the data based on the 'PREDICTED_CLASS' column (0 or 1)
                data_class_0 = df[df['PREDICTED_CLASS'] == 0]
                data_class_1 = df[df['PREDICTED_CLASS'] == 1]

                # Plot histograms with conditional coloring using 'hue'
                sns.histplot(data=df, x=selected_feature_comp, kde=False, bins=20, hue='PREDICTED_CLASS',  element="step", palette={0: '#2ca02c', 1: '#d62728'})

                # Add a vertical line for the selected client
                plt.axvline(client_info.iloc[0][selected_feature_comp], color='black', linestyle='dashed', linewidth=3, label="Selected Client")

                # Create custom legend entries with color swatches
                import matplotlib.patches as mpatches
                legend_handles = [
                    mpatches.Patch(color='#2ca02c', label="Approved"),
                    mpatches.Patch(color='#d62728', label="Rejected"),
                    plt.Line2D([0], [0], color='black', linestyle='dashed', linewidth=3, label="Selected Client")
                ]
                
                plt.legend(handles=legend_handles)

                st.pyplot(plt)