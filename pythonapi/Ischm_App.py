#!/usr/bin/env python
# coding: utf-8

# In[9]:


from flask import Flask, request, redirect, url_for, flash, jsonify, make_response
import pandas as pd
import numpy as np
import pickle
import json
import shap
from helper import *

# read pickle files
with open('score_objects.pkl', 'rb') as handle:
    d, features_selected, Extra1, explainer = pickle.load(handle)

Ischm_App = Flask(__name__)

@Ischm_App.route('/api/', methods=['POST'])
def makecalc():

    json_data = request.get_json()
    #read the real time input to pandas df
    data = pd.DataFrame(json_data)
    #transform DataFrame
    data = transform_categorical(data, d, features_selected)
    #score df
    prediction, probability = score_record(data, Extra1)
    #convert predictions to dictionary
    data['prediction'] = prediction
    data['probability'] = probability
    output = data.to_dict(orient='rows')[0]
    #output['plot'] = p
    return jsonify(output)

if __name__ == '__main__':

    Ischm_App.run(debug=True, host='0.0.0.0', port=5000)

