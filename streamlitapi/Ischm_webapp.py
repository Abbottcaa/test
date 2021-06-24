#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import requests
import datetime
import shap
import json
import pickle
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.title('Real Time Ischemic Event Prediction')

# Python API endpoint
url = 'http://pythonapi:5000'
endpoint = '/api/'

# description and instructions
st.write('''Real Time Ischemic Event Prediction.''')

st.sidebar.header('User Input features')

def user_input_features():
    input_features = {}
    input_features["DAPT_new"] = st.sidebar.slider('Dapt Duration', 0, 1460)
    input_features["BL_PLT"] = st.sidebar.slider('Platelet', 0, 900)
    input_features["AGE"]= st.sidebar.slider('Age', 18, 100)
    input_features["CRCL"] = st.sidebar.slider('Creatinine Clearance', 0, 520)
    input_features["DS"] = st.sidebar.slider('Diameter Stenosis', 0, 100)
    input_features["BL_FBG"] = st.sidebar.slider('Baseline glucose',  0, 800)
    input_features["N_TRTLSN"] = st.sidebar.slider('Number of treated lesions', 1, 10)
    input_features["BMI"] = st.sidebar.slider('BMI', 0, 120)
    input_features["BL_HGB"] = st.sidebar.slider('Baseline Hemoglobin', 0, 50)
    input_features["RVD"] = st.sidebar.slider('RVD', 1, 12)
    input_features["BL_WBC"] = st.sidebar.slider('Baseline WBC', 0, 150)
    input_features["TSTNTLGH"] = st.sidebar.slider('Total stent length',  1, 170)
    return [input_features]


json_data = user_input_features()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# read pickle files
with open('score_objects.pkl', 'rb') as handle:
    d, features_selected, Extra1, explainer = pickle.load(handle)

# explain model prediction results
def explain_model_prediction(data):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(Extra1)
    # Calculate Shap values
    shap_values = explainer.shap_values(data)
    p = shap.force_plot(explainer.expected_value[1], shap_values[1], data)
    return p, shap_values

submit = st.sidebar.button('Get predictions')
if submit:
    results = requests.post(url+endpoint, json=json_data)
    results = json.loads(results.text)
    results = pd.DataFrame([results])

    st.header('Final Result')
    prediction = results["prediction"]
    probability = results["probability"]

    st.write("Prediction: ", int(prediction))
    st.write("Probability: ", round(float(probability),3))

    #explainer force_plot
    results.drop(['prediction', 'probability'], axis=1, inplace=True)
    results = results[features_selected]
    p, shap_values = explain_model_prediction(results)
    st.subheader('Model Prediction Interpretation Plot')
    st_shap(p)
    

    st.subheader('Summary Plot 1')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[1], results)
    st.pyplot(fig)
    
    st.subheader('Summary Plot 2')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[1], results, plot_type='bar')
    st.pyplot(fig)

