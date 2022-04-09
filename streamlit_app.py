import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.title("Data Scientist Career Accelerator")
def load_data():
    """
    Write 1-2 lines of code here to load the data from CSV to a pandas dataframe
    and return it.
    """
    pass

st.header("Annual Salary Prediction")

col1,col2 = st.columns(2)
with col1:
    X = []
    tmp = st.checkbox('Python')
    X.append(int(tmp))
    tmp = st.checkbox('Spark')
    X.append(int(tmp))
    tmp = st.checkbox('AWS')
    X.append(int(tmp))


predict = st.button("PREDICT")

with col2:

    con = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Congratulation!</p>'
    st.markdown(con, unsafe_allow_html=True)
    #st.write("Congratulation!")
    st.write("You could make ")
    if predict:
        st.write("$1000000")
    else:
        st.write("_ _ _ _")
    st.write("a year!")



