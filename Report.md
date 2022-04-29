# Final Project Report

**Project URL**: https://share.streamlit.io/cmu-ids-2022/final-project-shueyrhonrhon/main
***Note: our salary predictions features runs slowly on this link, please run this project locally to use this feature.***

**Video URL**: TODO

## Introduction
Data Scientist is one of the most popular jobs in the United States. According to Harvard Business Review, Data Scientist is the sexist job of the 21st Century. On February 2, 2022, recruiting website Glassdoor released its annual ranking of the “50 Best Jobs in America for 2022”, and Data Scientist is the #3 Best Job in America, with a Median Base Salary of $120,000. In the Carnegie Mellon community, we also have met many people who are preparing for or curious about being a Data Scientist after graduating. However, some of them are still confusing what skills they need to have in order to get an ideal salary in the future. In this project, we want to present a clear picture of the salary distribution in the United States, provide guidelines to the Data Scientist applicants, and help them land a satisfying job. This project will give users a brief sense of what kind of skills they need to possess or know about. Users like students in Carnegie Mellon can register for courses for skills they want to possess.

First of all, we want to help our users to have a brief understanding of the salary distribution across all the states in the United States. They will know about the industries that companies belong to and the sizes, ownership, and glassdoor ratings of these companies. After having a brief sense of a Data scientist career in the United States, the project will provide a chart related to the importance of skills based on our machine learning model. The users can know what kind of skills they need to get a higher salary. At last, the user can choose to predict their salary based on state, industry, and sector options. The users then will know whether they can get an ideal job or not.

## Related Work
The dataset is sourced from Kaggle (https://www.kaggle.com/nikhilbhathi/data-scientist-salary-us-glassdoor), and it is made by scraping “Data Scientists” job postings in the United States from www.glassdoor.com. The dataset has a size of 3.12MB with 742 samples, and each sample has 42 columns. Specifically, those columns include: index, Job Title, salary Estimate, Job Description, Rating, Company Name, Location, Headquarters, Size, Founded, Type of ownership, industry, Sector, Revenue, Competitors, Hourly, Employer-provided, Lower Salary, Upper Salary, Avg Salary, company_txt, Job Location, Age, skills, job_title_sim, senior by title, and Degree.

Previous works of literature have explored the Data Scientist salaries. Situ, W.(et. al) **[1]** used gradient boosting decision tree and logistic regression to predict the probability to get a job in different categories of companies with expected salary mean. They collected about 3000 profiles from Linkedin and Glassdoor. They hope their work can help data science professionals to determine which rank of a company they have the greatest chance to enter and help the Human Resource department to understand the trends in the talent market when they need to build a competitive talent pool. In another survey conducted by King, J., & Magoulas, R. in 2015 **[2]**, they gave a worldwide data-science related salary and ease of finding a new position. We can find that the United States has the highest salary in the world. The average salary of a Data Scientist reaches more than 100k dollars a year in the United States. This means that people who want to pursue a Data Science career in the United States will have a higher possibility to get an ideal job with good pay. Also, it's not difficult to find a new position in a Data Science career since only 13% percent of respondents thought it was not easy. And people who believe that it's very easy to get a job even have a higher average salary. We believe that as long as people try their best to study data science skills and search for their jobs, they will get a good result.

Inspired by the above work, we believe that there are plenty of chances to get a Data Scientist job in the United States and it will not be a very difficult job. As a result, we want to use the nearest data to explore how to get an ideal job as a Data Scientist in the United States. We hope that our project will help people who want to start or make progress in their Data Scientist careers.
![related_work1](/images/related_work1.jpg)

![related_work2](/images/related_work2.jpg)

## Methods

An explanation of the techniques and algorithms you used or developed.

### Data Exploration
In this part, we use various interactive visualization techniques to allow users to freely explore the data and gain insights in the process. First, to give an overview of the geographical distribution of the salary of data scientists, we use a map of the US where the color encodes the average salary of each state. The interactive tooltips let users further explore detailed numbers. The map allows users to quickly get to know which locations offer data scientists jobs with higher pay. Second, for each specific state, we allow users to explore more details about the data slice. Users can select a state that they are interested in. And several visualizations will be generated to help them know about the job market in the state, including: 
1. A bar chart comparing the nationwide salary distribution and state salary distribution. It intuitively shows how many job opportunities exist for each pay range. And users can get a sense of how high the pay is generally in the state compared with the whole US.
2. A bar chart showing the top 5 industries that offer the highest average salary for data scientists. Users can use this chart to identify which industries they can work for in order to get a higher salary.
3. A scatter plot showing the distribution of company rating and salary. Users can get an overview of the data points regarding company ratings and salary.
4. Two pie charts showing the composition of company size and company ownership. Users can learn about what kind of companies are offering data scientist jobs in this state.

In the implementation of this part, we use techniques such as:
1. Color encoding. A sequential color palette is composed of varying intensities of a single hue of color at uniform saturation. Variability in luminance of adjacent colors corresponds to the variation in data values that they are used to render. The map uses a sequential color palette to encode the value of average data scientist jobs in each state. It is intuitive and easy for readers to gain a general view of the distribution.
2. Compare and contrast. Contrast graphics communicate two concepts or datasets side-by-side. This data visualization technique when you want the readers to gain information from the comparison. In our case, we want the readers to have a baseline to refer to when they look at the distribution of state salary. The bar chart contrasts state data with national data by stacking bars of different colors so users can have a better baseline reference.
3. Multi-view coordination. Interactive visualization allows users to freely explore the dataset. In this process, there are many pieces of information we want to present as multiple views to users. However, such views are not isolated and separated but rather in a close-tied relationship. In our implementation, when users choose a state, multiple charts update to visualize the data slice.
4. Details on demand. Sometimes users want to dive deep and get more details in the data they are interested in. However, presenting all the details statically will result in the view being too crowded with information, and making it hard to generate a general view and identify patterns. In the interactive form, we can offer details when users interact and request for them. In our implementation, when users hover their mouse over the chart, tooltips help users get to know more information.


### Salary Prediction
In this part, we experimented with various machine learning models to find the one with best performance. 
#### XGBoost
XGBoost, short for extreme gradient boosting, is an implementation of gradient boosted decision trees designed for speed and performance by Tianqi Chen **[3]**. XGBoost adds a regular term to control the complexity of the model, which helps to prevent over fitting and improve the generalization ability of the model. It uses the second-order Taylor expansion to find the optimal solution. The XGBoost model in this paper is implemented by the xgbregressor() function in Python's xgboost library, with a max depth of 7 and a learning rate of 0.1. 

This algorithm has recently been dominating applied machine learning and Kaggle competitions. Not surprisingly, it achieved the best performance among the machine learning models we have experimented with, with a mean cross validation score 0.2 higher than the others. Thus, we decided to use a trained XGBoost regressor as our model. 

#### Random Forest
Random forest regressor refers to a model that uses multiple decision trees to train and predict samples. The method was first proposed by Leo Breiman and Adele Cutler **[4]**, generally thought as an embodiment of bagging in ensemble learning. Bagging is a representative method of parallel integrated learning method based on bootstrap sampling. Given a standard data set t of size n, bagging randomly and evenly samples m rounds from t to get m sample sets into the sampling set, and then puts the samples back into the initial data set. By replacing sampling, some results can be observed repeatedly in each subset, and some samples in the initial training set will never appear in the sampling set, These unseen samples can be used as validation sets to estimate the subsequent generalization performance. Based on each sampling set, a base learner can be trained, and then the results of all base learners can be integrated. Usually, the results of regression tasks are combined with simple averaging method. 

In random forest, the decision tree is the base learner for training based on bagging algorithm. On this basis, the method also introduces the selection of random features in the training process of the decision tree. For each node of the base decision tree, a subset containing some features will be randomly selected from the feature set of the node, and then an optimal feature will be selected from this subset for division. Therefore, the basic learning machine of random forest has disturbances from samples and characteristics. This diversity further enhances the generalization ability of the final method integration. The random forest model in this project is implemented by Python's kneighborsregressor() function. However, it failed to achieve the optimal performance therefore was not selected in the final code of our project.

#### K Nearest Neighbor (KNN)
KNN was first proposed by Cover and Hart in 1968 **[5]**. Among all machine learning methods, the core idea of KNN is the most intuitive, it uses feature similarity to predict the values of any new data points. This means that the new point is assigned a value based on how closely it resembles the points in the training set. The average of these data points is the final prediction for the new point. 

The algorithm steps are as follows: firstly, according to the given distance measurement method (Euclidean distance is used in this project), find the K sample points closest to the sample points in the training set. Then, the category of sample points is determined according to the average value in this group. For the value of K, there is no fixed formula, and the most suitable number of samples needs to be found through parameter tuning experiment. Generally speaking, the decrease of K value will make the overall model more complex and prone to over fitting; The K value increases, that is, the neighborhood of the training sample is expanded. Although the training error will increase, the generalization performance will be improved. KNN algorithm is fast and can undertake the regression task of data with large sample size. However, it was not preferrable for our project as we only have hundreds of samples in our dataset, and the performance was not satisfying. Thus, we eventually rule this method out of our project implementation.

#### Feature Importance
From the section mentioned above, we eventually selected a trained and tuned XGBoost Regressor as our final model. A benefit of using this kind of gradient boosting method is that after the boosted trees are constructed, it is relatively straightforward to retrieve importance scores for each attribute.

Feature importance provides a score that indicates how useful or valuable each feature was in the construction of the boosted decision trees within the model. The more an attribute is used to make key decisions with decision trees, the higher its relative importance.This importance is calculated explicitly for each attribute in the dataset, allowing attributes to be ranked and compared to each other. It is measured by the amount that each attribute split point improves the performance measure, weighted by the number of observations the node is responsible for, then averaged across all of the the decision trees within the model **[6]**.

A trained XGBoost model could automatically calculates feature importance on the training dataset. These importance scores are available in the feature_importances_ member variable of the trained model. The scores could then be ranked and presented to users as a bar chart.

## Results

The data exploration part is aimed for users to gain an overview of the distribution of average salary around the United States. In this process, users can figure out what are the potential locations they want to work at in the future. Users can then select the state they are interested in to get more details from a set of generated plots. Users can see how many jobs there are for each salary range. Users can see what kind of industries there exist that offer well-paid job opportunities and whether it aligns with their interest and past experience. Users can also see the composition of company ratings, sizes and ownerships. After the data exploration, users will identify several potential locations and industries and proceed to the salary prediction part.

Our salary prediction feature is aimed to help applicants address two problems in real life. The first problem is career exploration: we aim to help college students who are looking for an internship / entry-level job as a data scientists choose their future career, by showing them the expected salary through our machine learning model. The second problem is interview preparation: “What is your expected salary” is a common question that interviewer asks, however, it is hard for an unexperienced college students to answer this question. Through our prediction feature, users could have a general sense on the expected salary of people who have similar backgrounds. 
Specifically, here are the options for users to choose :

* ***Location***: users are allowed to choose from all states in the United States.
* ***Industry***: Industry like IT industry, Pharmaceutical industry etc
* ***Sector***: Sector in which company works
* ***Company Size***: Range of number of employee working in the company
* ***Type of Ownership***: if the company is private, public or government owned
* ***Seniority***: Seniority in title, users are able to see the growth path of the job.
* ***Degree***: Master, or Phds. 
* ***Rating***: The corresponding rating of the company. 


## Discussion
From the data exploration part, we can gain some insights on the distribution of average salary. From the map we can see that California and Illinois have the highest average salaries in the US. It makes sense because there are lots of data scientists working in technology industries in California and technology companies often offer higher pay. And for Illinois, there are lots of Fintech companies in Chicago that offer top salaries to data scientists to research on trading strategies. We can also see that a lot of states in the middle west do not have any samples of data scientists jobs because of the scarcity.

We also select specific states to create a case study.
1. California. We can see from the stacked bar plot that California offers more well paid job opportunities than the national average. The top industries are mostly IT and software related. And most companies are middle-sized private companies.
2. Texas. Salaries of job offers in Texas are average. The top industries are mostly traditional industries. And most companies are big private companies.

There are some interesting insights we could conclude from the salary prediction model.
1. No strong correlation exists between the 16 skills (Python, Spark, etc.) and the income salary alone. The mean cross-validation score for machine learning model could only reach 0.3 when only these 16 features are considered. 
2. Besides Python, which is probably the most important and popular skills in the world of data science, we found out the Google Analytics (A web analytics service that provides statistics and basic analytical tools for search engine optimization and marketing purposes **[7]**) and SAS (A command-driven software package used for statistical analysis and data visualization **[8]**) also play an important role in data scientists salary. This reminds future data scientists to expand and update their skill sets to meet job market demand.
3. Industrial differences exists when it comes to data scientists' salary. Although having similar job description and same job title, some industries such as business, IT and retail could pay much more than other industries. 

## Future Work
Currently, we only have no more than 800 pieces of data from glassdoor. These data can not represent the salary distribution across the US, not to mention the world. We hope that we can get more data from other resources like Linkedin, Indeed, and other applications from other countries. We believe that our application can be more persuasive with more data. Also, we don't provide all the summarized information for our users. For example, the degree requirement can not be visualized because we have too many 'na' in our data. We believe that we can provide more summarized information with more data, which will give our users a more clear vision of the salaries of Data scientists.

On the other hand, we provide limited functions for our users. Currently, we can only predict salary based on the users' choices based on 9 choices, like location, industry, and sector. We hope that in the future we can provide more choices like no-tech skills that users possess, the number of years he/she has worked and their previous salary. These data will be collected in the future to further train our machine learning model so that we can give a more precise prediction. 

For our models specifically:
* Model Accuracy:  In order to increase the accuracy of model, more features and training data in longer time series are required. For example, for the degree option, the current didn’t include options for high school and bachelor’s degrees. 

* Model Interpretation: We use Xgboost to train our model,  although it have significant higher accuracy than others, it is not a very interpretable model. It will have two main drawbacks: 1. the overall user experience will be reduced because we cannot directly tell them how to improve the objective, instead, they have to see the difference trying different options by themselves. 2. It lowers the credibility of our model. However, we believe if the training size could be increase, we will shift to other more interpretable models with acceptable accuracy levels.

* Intrinsic Bias: Since our data were scrapped from the job posting in Glassdoor, it contains the bias that the real working environment != to the job descriptions. For example, the hiring manager wrote the role require MBAs degree, but in real scenario, a undergraduate or master students could still get the job, as long as they obtain the required abilities; The expected salary could be varies to each person working in this role, because there will be a threshold for negotiation. To improve the performance of our model, we need to sample more data from the real working environment. 


## Reference
[1] Situ, W., Zheng, L., Yu, X., & Daneshmand, M. (2017). Predicting the probability and salary to get data science job in top companies. IIE Annual Conference.Proceedings, , 933-939.

[2] King, J., & Magoulas, R. (2015). 2015 data science salary survey. O'Reilly Media, Incorporated.


[3] Chen, T., & Guestrin, C. (2016, August). Xgboost: A scalable tree boosting system. In *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining* (pp. 785-794).

[4] Breiman, L. (2001). Random forests. *Machine learning*, *45*(1), 5-32.

[5] *K-Nearest Neighbors Algorithm: KNN Regression Python*. Analytics Vidhya. (2020, May 25). Retrieved April 28, 2022, from https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/ 

[6] Brownlee, J. (2020, August 27). *Feature importance and feature selection with XGBoost in python*. Machine Learning Mastery. Retrieved April 28, 2022, from https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/ 

[7] Chai, W., & Contributor, T. T. (2021, April 12). *What is google analytics and how does it work?* SearchBusinessAnalytics. Retrieved April 28, 2022, from https://www.techtarget.com/searchbusinessanalytics/definition/Google-Analytics#:~:text=Google%20Analytics%20is%20a%20web,anyone%20with%20a%20Google%20account. 



[8] *Statistical & Qualitative Data Analysis Software: ABOUT SAS*. LibGuides. (n.d.). Retrieved April 28, 2022, from https://libguides.library.kent.edu/statconsulting/SAS 
