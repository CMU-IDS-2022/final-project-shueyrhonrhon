from sklearn.preprocessing import scale
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
# from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVR

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

def Xgb_Regression():
    # model = xgb.XGBRegressor()
    model = xgb.XGBRegressor(max_depth=7, eta=0.1,n_estimators=1000, learning_rate=0.1, use_label_encoder=False)
    return model
def Svm_regression():
    model = SVR(kernel='rbf')
    return model


def train_model(x,y):
    #model = Xgb_Regression()
    model = Svm_regression()
    print(x.shape)
    print(y.shape)
    # xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.2)
    xtrain = x
    ytrain = y
    model.fit(xtrain, ytrain)
    # feature importance
    # print(model.feature_importances_)
    # #print(model.feature_importances_.T.shape)
    # # plot
    # # pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # # pyplot.show()
    # df = pd.DataFrame(model.feature_importances_,columns=['importance'])
    # df['skills'] = x.columns
    # #df.reindex(x.columns, axis=1)
    # print(df)
    # feature_chart = alt.Chart(df).mark_bar(color="#C0EDA6").encode(
    #   x=alt.X('skills',sort="-y"),
    #   y=alt.Y('importance'),
    # )
    # st.write(feature_chart)
    #   x=alt.X("Industry",sort="-y"),
    # y=alt.Y("mean(avgSalary)"),
    # tooltip = ['Industry', 'mean(avgSalary)']

    #evaluate
    scores = cross_val_score(model, xtrain, ytrain,cv=10)
    print("Mean cross-validation score: %.2f" % scores.mean())
    # ypred = model.predict(xtest)
    # mse = mean_squared_error(ytest, ypred)
    # print("MSE: %.2f" % mse)


Y = df.iloc[:,19]
#print(Y.head)
X = df.iloc[:,23:39]
#print(X.head)
train_model(X,Y)

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



