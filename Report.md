# Final Project Report

**Project URL**: https://share.streamlit.io/cmu-ids-2022/final-project-shueyrhonrhon/main

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
1. Color encoding. The map uses color encoding to present salary information intuitively.
2. Contrast. The bar chart contrasts state data with national data so users can have a better baseline reference.
3. Multi-view coordination. When users choose a state, multiple charts update to visualize the data slice.
4. Details on demand. When users hover their mouse over the chart, tooltips help users get to know more information.


## Results

## Discussion

## Future Work
Currently, we only have no more than 800 pieces of data from glassdoor. These data can not represent the salary distribution across the US, not to mention the world. We hope that we can get more data from other resources like Linkedin, Indeed, and other applications from other countries. We believe that our application can be more persuasive with more data. Also, we don't provide all the summarized information for our users. For example, the degree requirement can not be visualized because we have too many 'na' in our data. We believe that we can provide more summarized information with more data, which will give our users a more clear vision of the salaries of Data scientists.

On the other hand, we provide limited functions for our users. Currently, we can only predict salary based on the users' choices based on 9 choices, like location, industry, and sector. We hope that in the future we can provide more choices like no-tech skills that users possess, the number of years he/she has worked and their previous salary. These data will be collected in the future to further train our machine learning model so that we can give a more precise prediction. 

## Reference
[1] Situ, W., Zheng, L., Yu, X., & Daneshmand, M. (2017). Predicting the probability and salary to get data science job in top companies. IIE Annual Conference.Proceedings, , 933-939.

[2] King, J., & Magoulas, R. (2015). 2015 data science salary survey. O'Reilly Media, Incorporated.
