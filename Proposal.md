# Final Project Proposal

**GitHub Repo URL**: https://github.com/CMU-IDS-2022/final-project-shueyrhonrhon

**Data Scientist Career Accelerator**

Team Member : Zhiyi Li, Yining(Nora) Wang, Jiaxiang Wu, Ge Huang

**Project Goal**

Data Scientist is one of the most popular jobs in the US. According to Harvard Business Review, Data Scientist is the sexist job of the 21st Century. On February 2, 2022, recruiting website Glassdoor released its annual ranking of the “50 Best Jobs in America for 2022”, and Data Scientist is the #3 Best job in America, with a Median Base Salary $120,000. In this project, we want to present a clear picture of the salary distribution in the US, provide guidelines to the Data Scientist applicants, and help them land a satisfying job.

**Dataset Overview**

The dataset is sourced from Kaggle (https://www.kaggle.com/nikhilbhathi/data-scientist-salary-us-glassdoor), and it is made by scraping “Data Scientists” job postings in United States from www.glassdoor.com. The dataset has a size of 3.12MB with 742 samples, and each sample has 42 columns. Specifically, those columns includes: index, Job Title, salary Estimate, Job Description, Rating, Company Name, Location, Headquarters, Size, Founded, Type of ownership, industry, Sector, Revene, Competitors, Hourly, Employer provided, Lower Salary, Upper Salary, Avg Salary, company_txt, Job Location, Age, skills, job_title_sim, senior by title, and Degree.


**Project Plan**

Our project will be divided into two parts, and each two of our group member will be in charge of one part: 

**Data Filter**

We want to analyze our data in different dimensions. For example, the distribution of   salary based on the locations of companies. The users will be able to see the basic salary distribution on a US map, and we will provide multiple selection boxes for users to choose their parameters and use different kinds of charts to show the distribution of salary data. We hope these charts will give the user a clear picture of salary distribution in the United States.

**Relationship between Salary and Skills, and model predictions**

The dataset contains the skill set requirements for each job posting. The skills include Python, Spark, aws, excel, sql, sas, keras, pythorch, scikit, tensor, hadoop, tableau, bi, flink mongo and google_an. With those data, we can give an overview of what are the most popular and high demanding skills that a Data Scientist is expected to acquire. Furthermore, we can build machine learning predictive models to predict the expected salary range by letting users enter their current skill sets. Furthermore, for people who want to gain more knowledge to land a job with higher salary, we will build models that show the impact of each skill on salary growth. Thus, applicants will not waste their time learning skills that have little impact on their salary.

**Sketches and Data Analysis**
1. Data Processing\
The data are well cleaned and processed. We have 42 parameters in total and each of them has 742 rows of data. Some data are converted into 0/1, which are very suitable for our machine learning model. We want to provide a summary about the distribution of   salary based on the locations of companies to the user. And user will choose what kind of skills he/she has and what states he/she wants to work in. Then we will give a recommendation of what kinds of skills he/she needs to work in those states. The following charts are parts of what we have explored:
**The distribution of average salary of data scientists
(x axis is average salary, y axis is count)**
![](/images/national_avg_salary.png)
**The average salary across the US**(The redder the greater the value)

![a map](/images/nation_salary_map.png)

\
In order to build a machine learning model, we need to preprocessing the y-labels, which is the expected salary estimates. In the raw data, the salary is represented in ranges(as shown below), and we need to preprocessing it into numerical values, by calculating its mean values. 
![A sketch](/images/sketch_map.png)

2. System Design
\
First, we train a machine learning model using our dataset to determine the importance of each skill for landing a high salary job. As shown in the picture below:
![A sketch](/images/model-sketch.png)

\
After training our machine learning model with the dataset, we provide a fun interactive function where the users can select the skills they have, and we make a prediction of their annual salary based on this information. As shown in the picture below:


![A sketch](/images/model_prediction_sketch.png)
