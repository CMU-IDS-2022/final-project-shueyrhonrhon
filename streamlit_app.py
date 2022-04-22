from matplotlib.pyplot import legend
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
from sklearn.decomposition import PCA

from vega_datasets import data

st.title("Data Scientist Career Accelerator")
@st.cache
def load_data():
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

counties = alt.topo_feature(data.us_10m.url, 'counties')
source = data.unemployment.url


us_states = ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CQ', 'CT', 'DC', 'DE', 'FL', 'GA', 'GU',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY']
dic = {}
for i, s in enumerate(us_states):
    dic[s] = i + 1
df_map = df.groupby('Job Location').agg({'avgSalary': np.mean}).reset_index()
df_map['id'] = df_map['Job Location'].apply(lambda x : dic[x])

states = alt.topo_feature(data.us_10m.url, feature='states')

map_salary = alt.Chart(states).mark_geoshape().encode(
    color='avgSalary:Q'
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(df_map, 'id', ['avgSalary'])
).project('albersUsa')

background = alt.Chart(states).mark_geoshape(
    fill='lightgray',
    stroke='white'
).properties(
    title='US State DS salary',
    width=700,
    height=400
).project('albersUsa')

st.write(background + map_salary)


st.text("This app gives you a brief overview of the salary of Data Scientist in the US")
st.text("You can choose what skills you have and we will predict the salary!!!")



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

st.write(national_salary_chart+state_salary_chart)


st.header("Top 5 industry average salary in "+stateOption)
top_industry_salary_chart = alt.Chart(df[slices][:6]).mark_bar(color="#4D77FF").encode(
    x=alt.X("mean(avgSalary)"),
    y=alt.Y("Industry",sort="-x"),
    tooltip = ['Industry', 'mean(avgSalary)']
)
st.write(top_industry_salary_chart)

def Xgb_Regression():
    # model = xgb.XGBRegressor()
    model = xgb.XGBRegressor(max_depth=7, eta=0.1,n_estimators=1000, learning_rate=0.1)
    return model
def Svm_regression():
    model = SVR(kernel='rbf')
    return model
def pca_reduction(X):
    pca = PCA(n_components=3)
    pca.fit(X)
    X_new = pd.DataFrame(pca.transform(X))
    print(X_new.head)
    return X_new

def train_model(x,y):

    #print(x.head)
    model = Xgb_Regression()
    #model = Svm_regression()
    #print(x.shape)
    #print(y.shape)
    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.1)
    #xtrain = x
    #ytrain = y
    model.fit(xtrain, ytrain)
    #feature importance
    #print(model.feature_importances_)
    #print(model.feature_importances_.T.shape)
    # plot
    # pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # pyplot.show()
    df = pd.DataFrame(model.feature_importances_,columns=['importance'])
    df['skills'] = x.columns
    df.reindex(x.columns, axis=1)
    #print(df)
    feature_chart = alt.Chart(df).mark_bar(color="#C0EDA6").encode(
      x=alt.X('skills',sort="-y"),
      y=alt.Y('importance'),
    )
    st.write(feature_chart)

    #evaluate
    scores = cross_val_score(model, xtrain, ytrain,cv=5)
    print("Mean cross-validation score: %.2f" % scores.mean())
    ypred = model.predict(xtest)
    mse = mean_squared_error(ytest, ypred)
    print("MSE: %.2f" % mse)

def expand_data(df):
    # size = []
    # for (idx, row) in df.iterrows():
    #     if '-' in row.loc['Size']:
    #         size.append(row['size'].split(' - ')[1])
    #     else:
    #         size.append(row['size'])
    # print(size)
    size = []
    for i in range(len(df)):
        if ' - ' in df.loc[i, "Size"]:
            #print(df.loc[i, "Size"].split(' - '))
            tmp = df.loc[i, "Size"].split(' - ')[1]
        elif '-' in df.loc[i, "Size"]:
            tmp = df.loc[i, "Size"].split('-')[1]
        elif '+' in df.loc[i, "Size"]:
            tmp = df.loc[i, "Size"].split('+')[0]
        else :
            tmp = df.loc[i, "Size"]
        #print(tmp)
        if 'unknown' in tmp:
            size.append(1)
        else :
            size.append(int(tmp))
    df['new_size'] = size
    #print(df['new_size'])
    df = df.loc[df.index.repeat(df.new_size)].reset_index(drop=True)
    return df

#df = expand_data(df)
Y = df.iloc[:,19]
#print(Y.head)

X = pd.concat([df.iloc[:,23:39],df.iloc[:,21],df.iloc[:,12],df.iloc[:,40:42]],axis = 1)
#X = pd.concat([df.iloc[:,21],df.iloc[:,12]],axis = 1)
#X = df.iloc[:,23:39]
X["Job Location"] = X["Job Location"].astype("category")
X["Sector"] = X["Sector"].astype("category")
X["seniority_by_title"] = X["seniority_by_title"].astype("category")
X["Degree"] = X["Degree"].astype("category")
X["Job Location"] = X["Job Location"].cat.codes
X["Sector"] = X["Sector"].cat.codes
X["seniority_by_title"] = X["seniority_by_title"].cat.codes
X["Degree"] = X["Degree"].cat.codes
#X = pca_reduction(X)
#print(X.head)

train_model(X,Y)

st.header("Annual Salary Prediction")
col1,col2 = st.columns(2)
with col1:
    skill_set = df.columns[23:39].values
    X = []
    for skill in skill_set:
        tmp = st.checkbox(skill)
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



