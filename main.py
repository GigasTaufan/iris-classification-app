# IMPORT LIBRARIES
from turtle import width
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# SET PAGE CONFIG
st.set_page_config(page_title="Iris Classification", page_icon=":bouquet:", layout="wide")

# THE TITLE
st.title(":bouquet: Iris Flower Classification")

st.write("""
    ### Simple Iris Flower Prediction App

    This web app predict the Iris flower type
""")

# SIDEBAR
## User can input the parameter here
st.sidebar.header('User Input Parameter')

## Function to collect the parameter 
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

## Function for prediction
def prediction_type(prediction):
    st.write("""
        <style>
        .big-font {
            font-weight:bold !important;
            color: #F675A8;
            font-size: 25px;
        }
        </style>
    """, unsafe_allow_html=True)


    if prediction == 'setosa':
        st.write('<p class="big-font">Setosa</p>', unsafe_allow_html=True)
    elif prediction == 'versicolor':
        st.write('<p class="big-font">Versicolor</p>', unsafe_allow_html=True)
    else:
        st.write('<p class="big-font">Virginica</p>', unsafe_allow_html=True)


# MAINPAGE
## Call the user_input_feature to collect the parameters into a dataframe
df = user_input_feature()

## Show the values of the the parameters
st.subheader('User Input Parameters')
st.write(df)

# LOAD THE IRIS DATASET
iris = datasets.load_iris()
x = iris.data
y = iris.target

## Call the Random Forest Classification Method
clf = RandomForestClassifier()
## Train the model
clf.fit(x, y)
## Predict the clasification from the parameters values
prediction = clf.predict(df)
## The probability of the classification
prediction_probability = clf.predict_proba(df)


# SHOW THE PREDICTION AND THE PROBABILITY OF THE PREDICTION
st.subheader("### Class labels and their corresponding index number")
st.write(iris.target_names)

st.subheader('### Prediction')
prediction_type(iris.target_names[prediction])

st.subheader("### Prediction Probability")
st.write(prediction_probability)









