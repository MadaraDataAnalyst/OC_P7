# Prêt à dépenser - Default Risk Calculator Project

## Introduction
Welcome to the *Prêt à dépenser* - Default risk calculator project, part of the OpenClassrooms Data Scientist diploma Project 7: *Implémentez un modèle de scoring.*

## Project Overview
This project aims to develop a credit scoring tool to calculate the probability of a client repaying their credit and classify applications as approved or denied. To meet the increasing demand for transparency in credit decisions, an interactive dashboard has been created. This tool empowers relationship managers to transparently explain credit decisions and allows clients easy access to explore their personal information.

## Project Structure
- **mlflow_model:** Directory containing the MLflow artifacts and model.
- **streamlit_cloud:** CSV file for Streamlit Cloud.
- **.gitignore:** File specifying which files and directories to ignore in version control.
- **LGBMClassifier_with_nulls_model.joblib:** Pre-trained model saved as a joblib file.
- **LOGO.png:** Logo image used in Streamlit app.
- **P7_modeling.ipynb:** Jupyter notebook with data cleaning, feature engineering, and modeling.
- **api_test.py:** Unit tests for the API.
- **app.py:** Flask API code.
- **data_drift_report.html:** HTML report on data drift.
- **requirements.txt:** File specifying required packages and their versions.
- **shap_values.pkl:** Pickle file containing SHAP values.
- **streamlit.py:** Streamlit application code.

## Dashboard 
Explore the interactive dashboard: [Default Risk Calculator Dashboard](https://default-risk-calculator.streamlit.app/) 

The dashboard functions as a web application hosted on the Streamlit Share cloud platform, and its source code is integrated with this GitHub repository. Simultaneously, the API is deployed on the PythonAnywhere cloud platform. When the dashboard initiates requests, the API promptly provides clients' data along with predictions regarding the likelihood of loan repayment.

## Credits
- Data used: [Home Credit Default Risk Kaggle Dataset](https://www.kaggle.com/c/home-credit-default-risk/data)
- Kernel used for feature engineering: [LightGBM with Simple Features](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script)


