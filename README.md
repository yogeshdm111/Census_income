# Adult Census Income Prediction

This project uses the Adult Census Income dataset to build a machine learning model that predicts whether an individual's income exceeds $50K/year based on demographic data. The project includes a Jupyter Notebook for model training and a Streamlit web app for model deployment and sample predictions.

## Project Structure

- `adult-census-income-prediction.ipynb`: Jupyter Notebook for data preprocessing, model training, and evaluation.
- `streamlit_app.py`: Streamlit app script for deploying the trained model and making sample predictions.
- `XGBClassifier.pkl`: Serialized (pickled) XGBoost classifier model.
- `scaler.pkl`: Serialized (pickled) StandardScaler used for data normalization.
- `adult.csv`: The dataset used for training and testing the model.

## Requirements

- Python 3.11 or higher
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `streamlit`

You can install the required packages using:
```bash
pip install pandas numpy scikit-learn xgboost streamlit

## Running the Jupyter Notebook

-- Ensure you have Jupyter Notebook installed. If not, you can install it using following command  or you can use anaconda:
    pip install notebook
-- Open the Jupyter Notebook:
    adult-census-income-prediction.ipynb
-- Execute the cells in the notebook to preprocess the data, train the model, and save the trained model and scaler as pickle files.

## Running the Streamlit App

To run the Streamlit app for making predictions, follow these steps:

In VS code open your project folder,open terminal

--Ensure you have Streamlit installed. If not, you can install it using:

  pip install streamlit
-- Run the Streamlit app (stremlit file) using following command:
    streamlit run streamlit_app.py
The app will open in your default web browser. You can use the app to preview the dataset and make sample predictions based on user inputs.

Sample Prediction
The Streamlit app allows you to input sample demographic data and predict whether the income exceeds $50K/year. The model uses the XGBoost classifier trained on the dataset.

## Model Details
The project trains several models, including:
XGBoost Classifier
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
K-Nearest Neighbors Classifier
The best-performing model (XGBoost) is saved and used in the Streamlit app for predictions.







