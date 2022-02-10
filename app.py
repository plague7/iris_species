# 1. Library imports
import pandas as pd
import numpy as np
import streamlit as st
import os
import os.path
import joblib

from sklearn.neighbors import DistanceMetric
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

#model = IrisModel()

#Loading model locally
#model = joblib.load("C:/Users/Simplon/Desktop/Travaux python/API/Heroku Deploying/model.joblib")
#Functions :
MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'model.joblib')
with open(MODEL_DIR , 'rb') as handle:
    model = joblib.load(handle)

def predict_species():
    #global Sepal_Length, Sepal_Width, Petal_Length, Petal_Width
    data_in = [[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]]
    prediction = model.predict(data_in)
    probability = model.predict_proba(data_in).max()
    return st.write('This is the specie predicted:', str(prediction), 'And this is the score obtained:', probability,)
#
st.title('Welcome to the Iris Species classifier !')
st.header('Please write down the specifics of your Iris flower:')

# Writing the Data :
Sepal_Length = st.number_input('Sepal_Length', value=0.0, min_value = 0.0, max_value=10.0, step=0.01)
Sepal_Width  = st.number_input('Sepal_Width', value=0.0, min_value = 0.0, max_value=10.0, step=0.01)
Petal_Length = st.number_input('Petal_Length', value=0.0, min_value = 0.0, max_value=10.0, step=0.01)
Petal_Width  = st.number_input('Petal_Width', value=0.0, min_value = 0.0, max_value=10.0, step=0.01)


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
if st.button('Test'):
    predict_species()