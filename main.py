# IMPORT LIBRARIES
from turtle import width
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# SET PAGE CONFIG
st.set_page_config(page_title="Iris Classification", page_icon=":bouquet:", layout="wide")

st.write("""
    # Simple Iris Flower Prediction App

    This web app predict the Iris flower type
""")

# SIDEBAR
## User can input the parameter here
st.sidebar.header('User Input Parameter')

# Function to collect the parameter 
def user_input_feature():
    sepal_length = st.sidebar.slider('Sepal Lenght', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

    data = {
        'sepal_length':sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

    features = pd.DataFrame(data, index=[0])

    return features


