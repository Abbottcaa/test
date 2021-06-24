#!/usr/bin/env python
# coding: utf-8

# In[1]:


#helper functions – helper.py file
import shap
import pickle

# transform categorical data
def transform_categorical(data,d, features_selected):

    for i in list(d.keys()):
        data[i] = d[i].transform(data[i].fillna('NA'))
    return data[features_selected]

# score new data
def score_record(data, Extra1):

    return Extra1.predict(data)[0], Extra1.predict_proba(data)[:,1][0]

