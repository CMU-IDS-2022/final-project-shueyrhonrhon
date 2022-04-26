import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from streamlit_option_menu import option_menu
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from vega_datasets import data

st.set_page_config(layout="wide")
@st.cache()
def load_data(allow_output_mutation=True):
    data_path = "ds_data.csv"
    df = pd.read_csv(data_path)
    return df

def get_state_slices(df,state):
    lable = pd.Series([True]*len(df),index=df.index)
    if state:
        lable &= df['Job Location'].apply(lambda x : True if x==state else False)
    return lable

df = load_data()

def Xgb_Regression():
    model = xgb.XGBRegressor(max_depth=7, eta=0.1,n_estimators=1000, learning_rate=0.1)
    return model

@st.cache(allow_output_mutation=True)
def train_model(x,y):
    model = Xgb_Regression()
    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.1)
    model.fit(xtrain, ytrain)

    print("***")
    print(xtest)

    return model, xtrain, xtest, ytrain, ytest


def feature_importance(model,x):
    df = pd.DataFrame(model.feature_importances_,columns=['importance'])
    df['features'] = x.columns
    df.reindex(x.columns, axis=1)
    feature_chart = alt.Chart(df).mark_bar(color="#C0EDA6").encode(
      y=alt.Y('features',sort="-x"),
      x=alt.X('importance'),
      color="importance",
      tooltip = ['importance']
    ).interactive()
    st.altair_chart(feature_chart, use_container_width=True)

def model_accuracy(model, xtrain, xtest, ytrain, ytest):
    #evaluate
    scores = cross_val_score(model, xtrain, ytrain,cv=5)
    print("Mean cross-validation score: %.2f" % scores.mean())
    ypred = model.predict(xtest)
    mse = mean_squared_error(ytest, ypred)
    print("MSE: %.2f" % mse)

    y_pred_test = model.predict(xtest)

    source = pd.DataFrame({
        'ytest':ytest,
        'y_pred_test':y_pred_test,
    })

    predVSactual=alt.Chart(source).mark_circle(size=60).encode(
        x='ytest',
        y='y_pred_test',
    ).interactive()

    line = alt.Chart(source).mark_line(
        color='red',
        size=3
    ).encode(
        x="ytest",
        y="ytest",
    )
    st.altair_chart(predVSactual+line,use_container_width=True)

with st.sidebar:
    selected = option_menu(
        menu_title = "Data Scientist Career Accelerator",
        options = ["Data Exploration", "Salary Prediction"],
        default_index=0,
        icons=["boxes","search","clipboard-data"],
        menu_icon="people",
        styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "CornflowerBlue"},
    }
    )

    

if selected =="Data Exploration":
    st.title(f"You have selected {selected}")
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
        color='avgSalary:Q',
        tooltip=[
            alt.Tooltip("Job Location:N", title="State"),
            alt.Tooltip("avgSalary" + ":O", format='.1f', title="Average Salary"),
        ],
    ).transform_lookup(
        lookup='id',
        from_=alt.LookupData(df_map, 'id', ['Job Location', 'avgSalary'])
    ).project('albersUsa')

    background = alt.Chart(states).mark_geoshape(
        fill='lightgray',
        stroke='white'
    ).properties(
        title='map of average data scientist salary/K',
        width=700,
        height=400
    ).project('albersUsa')

    st.write(background + map_salary)


    st.header("Select the states you want to know about:")
    values = df['Job Location'].unique().tolist()
    default_ix = values.index('CA')
    state_option = st.selectbox(
            'state',
            values,
            index=default_ix
    )

    state_slices = get_state_slices(df,state_option)


    nationwide = pd.DataFrame()
    nationwide['avgSalary'] = df['avgSalary']
    nationwide['name'] = 'nationwide salary'
    state = pd.DataFrame()
    state['avgSalary'] = df[state_slices]['avgSalary']
    state['name'] = state_option + ' salary'

    category = ['0-50K', '50K-100K', '100K-150K', '150K-200K', '200K-250K', '250K-300K']
    nationwide['binned']=pd.cut(x=nationwide['avgSalary'], bins=[0,50,100,150,200, 250, 300], labels=category)
    nation_count = pd.value_counts(nationwide['binned']).reset_index()
    nation_count['name'] = 'nationwide salary'
    state['binned']=pd.cut(x=state['avgSalary'], bins=[0,50,100,150,200, 250, 300], labels=category)
    state_count = pd.value_counts(state['binned']).reset_index()
    state_count['name'] = state_option + ' salary'


    df_compare = nation_count.append(state_count, ignore_index=True)

    compare_chart = alt.Chart(df_compare, title = state_option + " v.s. nationwide").mark_bar(opacity=1.0).encode(
        x=alt.X('index', sort=category),
        y=alt.Y('binned:Q', stack=None),
        color="name",
    ).properties(
        width=700,
        height=300
    )


    top_industry_salary_chart = alt.Chart(df[state_slices][:6],title="top 5 industry in "+state_option).mark_bar(color="#4D77FF").encode(
        x=alt.X("mean(avgSalary)"),
        y=alt.Y("Industry",sort="-x"),
        tooltip = ['Industry', 'mean(avgSalary)'],
        color=alt.Color(
            "mean(avgSalary)" + ":Q",
            scale = alt.Scale(scheme="blues", reverse=False),
            legend=None,
        ),
    ).properties(
        width=700,
        height=300
    )


    st.write(compare_chart)
    st.write(top_industry_salary_chart)

    rating_chart = alt.Chart(df[state_slices], title = "rating and salary in "+state_option).mark_point(tooltip=True,size=80).encode(
        x=alt.X("avgSalary",scale=alt.Scale(zero=False)),
        y=alt.Y("Rating",sort="-x"),
    ).properties(
        width=700,
        height=300
    ).transform_filter(
        alt.datum.Rating != -1
    ).interactive()

    size_donut_chart = alt.Chart(df[state_slices],title = "company size in "+state_option).mark_arc(innerRadius=50,tooltip=True).encode(
        theta=alt.Theta(field="Size",aggregate="count"),
        color=alt.Color(field="Size"),
    ).properties(
        width=700,
        height=300
    )

    ownership_donut_chart = alt.Chart(df[state_slices],title = "Amount of different ownership companies in "+state_option).mark_arc(innerRadius=50,tooltip=True).encode(
        theta=alt.Theta(field="Type of ownership",aggregate='count'),
        color=alt.Color(field="Type of ownership"),
    ).properties(
        width=700,
        height=300
    )

    st.altair_chart(ownership_donut_chart)

    st.write(rating_chart)
    st.write(size_donut_chart)

if selected=="Salary Prediction":
    with st.sidebar:
        st.header("This code will be printed to the sidebar.")
    st.title(f"You have selected {selected}")
    st.header("What is the most important feature for salary increase?")
    Y = df.iloc[:,19]
    X = pd.concat([df.iloc[:,4],df.iloc[:,23:39],df.iloc[:,21],df.iloc[:,8],df.iloc[:,10],df.iloc[:,11],df.iloc[:,12],df.iloc[:,40:42]],axis = 1)
    print(X.head)
    X["Job Location"] = X["Job Location"].astype("category")
    X["Industry"]=X["Industry"].astype("category")
    X["Sector"] = X["Sector"].astype("category")
    X["seniority_by_title"] = X["seniority_by_title"].astype("category")
    X["Degree"] = X["Degree"].astype("category")
    X["Size"] = X["Size"].astype("category")
    X["Type of ownership"]=X["Type of ownership"].astype("category")

    X["Job Location"] = X["Job Location"].cat.codes
    X["Industry"] = X["Industry"].cat.codes
    X["Sector"] = X["Sector"].cat.codes
    X["seniority_by_title"] = X["seniority_by_title"].cat.codes
    X["Degree"] = X["Degree"].cat.codes
    X["Size"] = X["Size"].cat.codes
    X["Type of ownership"] = X["Type of ownership"].cat.codes


    model, xtrain, xtest, ytrain, ytest  = train_model(X,Y)
    feature_importance(model,X)
    model_accuracy(model, xtrain, xtest, ytrain, ytest)

    print(X.columns)

    ##Make predictions based on selection

    st.header("Predict your expected salary!")

    X_modified = pd.concat([df.iloc[:,4],df.iloc[:,23:39],df.iloc[:,21],df.iloc[:,8],df.iloc[:,10],df.iloc[:,11],df.iloc[:,12],df.iloc[:,40:42]],axis = 1)
    newRow = {i:0 for i in X.columns}

    with st.form("my_form"):
        job_location_option = st.selectbox(
            'Where do you want to be located?',
            (df['Job Location'].unique()))

        newRow['Job Location']=job_location_option

        industry_option = st.selectbox(
            'What industry you want to be in?',
            (df['Industry'].unique()))

        newRow['Industry']=industry_option

        sector_option = st.selectbox(
            'What sector you want to be in?',
            (df['Sector'].unique()))

        newRow['Sector']=sector_option

        size_option = st.selectbox(
            'What is your expected company size?',
            (df['Size'].unique()))

        newRow['Size']=size_option

        ownership_option = st.selectbox(
            'What is your expected type of ownership?',
            (df['Type of ownership'].unique()))

        newRow['Type of ownership']=ownership_option

        seniority_option = st.selectbox(
            'Are you looking forward to a senior role?',
            (df['seniority_by_title'].unique()))

        newRow['seniority_by_title']=seniority_option

        degree_option = st.selectbox(
            'What types of degree you acquired?',
            (df['Degree'].unique()))

        newRow['Degree']=degree_option

        rating = st.slider('Your expected company rating?', -1, 4, 5)
        newRow['Rating']=rating

        st.header("Check the skills you've acquired:")
        selected_skillsets = []
        

    # Every form must have a submit button.
        col1,col2,col3, col4 = st.columns(4)
        with col1:
            skill_set = df.columns[23:27].values
            for skill in skill_set:
                tmp = st.checkbox(skill)
                selected_skillsets.append(int(tmp))
        with col2:
            skill_set = df.columns[27:31].values
            for skill in skill_set:
                tmp = st.checkbox(skill)
                selected_skillsets.append(int(tmp))

        with col3:
            skill_set = df.columns[31:35].values
            for skill in skill_set:
                tmp = st.checkbox(skill)
                selected_skillsets.append(int(tmp))


        with col4:
            skill_set = df.columns[35:39].values
            for skill in skill_set:
                tmp = st.checkbox(skill)
                selected_skillsets.append(int(tmp))


        skills = ['Python', 'spark', 'aws', 'excel', 'sql', 'sas', 'keras',
                'pytorch', 'scikit', 'tensor', 'hadoop', 'tableau', 'bi', 'flink',
                'mongo', 'google_an']


        for i in range(len(skills)):
            newRow[skills[i]] = selected_skillsets[i]



        X_modified = X_modified.append(newRow,ignore_index=True)
        X_modified["Job Location"] = X_modified["Job Location"].astype("category")
        X_modified["Industry"]=X_modified["Industry"].astype("category")
        X_modified["Sector"] = X_modified["Sector"].astype("category")
        X_modified["seniority_by_title"] = X_modified["seniority_by_title"].astype("category")
        X_modified["Degree"] = X_modified["Degree"].astype("category")
        X_modified["Size"] = X_modified["Size"].astype("category")
        X_modified["Type of ownership"]=X_modified["Type of ownership"].astype("category")

        X_modified["Job Location"] = X_modified["Job Location"].cat.codes
        X_modified["Industry"] = X_modified["Industry"].cat.codes
        X_modified["Sector"] = X_modified["Sector"].cat.codes
        X_modified["seniority_by_title"] = X_modified["seniority_by_title"].cat.codes
        X_modified["Degree"] = X_modified["Degree"].cat.codes
        X_modified["Size"] = X_modified["Size"].cat.codes
        X_modified["Type of ownership"] = X_modified["Type of ownership"].cat.codes

        new_row_encode = pd.DataFrame(X_modified.iloc[[-1]])
        new_predicted = model.predict(new_row_encode)[0]

        predict_button = st.form_submit_button("Get My Salary Prediction")
    if predict_button:
        st.write("Your expected salary is ",new_predicted,"K for a year!")


