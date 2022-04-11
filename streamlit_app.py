from sklearn.preprocessing import scale
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.title("Data Scientist Career Accelerator")
@st.cache
def load_data():
    """
    Write 1-2 lines of code here to load the data from CSV to a pandas dataframe
    and return it.
    """
    data_path = "ds_data.csv"
    df = pd.read_csv(data_path)
    return df

@st.cache
def get_slice(df,state):
    lable = pd.Series([True]*len(df),index=df.index)
    if state:
        lable &= df['Job Location'].apply(lambda x : True if x==state else False)
    return lable


df = load_data()
if st.checkbox("Show raw data"):
    st.write(df[:25])

st.header("The salary map of Data Scientist in the US")


national_salary_chart = alt.Chart(df).mark_bar(color='#FF8080').encode(
    x = alt.X("avgSalary", bin=alt.Bin(extent=[0,300],step=50)),
    y=alt.Y("count()"),
    tooltip = ['avgSalary', 'count()']
)

stateOption = "CA"
slices = get_slice(df,stateOption)
st.header("The salary in "+stateOption+" and the US")
state_salary_chart = alt.Chart(df[slices]).mark_bar(color="#C0EDA6").encode(
    x=alt.X("avgSalary", bin=alt.Bin(extent=[0,300],step=50)),
    y=alt.Y("count()"),
    tooltip = ['avgSalary', 'count()']
)

st.header("Top industry average salary in "+stateOption)
top_industry_salary_chart = alt.Chart(df[slices]).mark_bar(color="#4D77FF").encode(
    x=alt.X("Industry",sort="-y"),
    y=alt.Y("mean(avgSalary)"),
    tooltip = ['Industry', 'mean(avgSalary)']
)

st.write(national_salary_chart+state_salary_chart)
st.write(top_industry_salary_chart)

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



