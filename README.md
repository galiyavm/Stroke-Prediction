# Stroke-Prediction
Introduction:
A stroke is a medical emergency. It happens when a blood vessel in the brain bursts or, more commonly, when a blockage happens. Without treatment, cells in the brain quickly begin to die. This can cause serious disability or death.Several risk factors believed to be related to the cause of stroke has been found by inspecting the affected individuals.Using this risk factors ,a number of works have been carried out for predicting and classifying stroke diseases.Most of the models are based on data mining and machine learning algorithms.In this work, we have used 3 machine learning algorithms to detect the type of stroke that possible occur or occurred from a person’s physical state and medical report data.

Objective:
I would like to identify the main drivers of stroke and make a model that 
would accurately predict stroke with in a month.

Attribute Information:
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
Note: "Unknown" in smoking_status means that the information is unavailable for this patient

Description of Dataset:

Here are 12 columns  and 5110 rows of data analysed the columns.
8 columns  are consisting of categorical data  while other 3 columns are continuous.
There were 201 missing values on body mass index(bmi) that I filled with mean  value.

Methodology:
1.Importing necessary libraries
2.Importing Dataset
3.Data cleaning
4.Feature Engineering
  a.Label Encoding
  b.Splitting Data for train and test
  c.Normalize using fit method
  d.Model Creation:Created few models namely, Logistic Regression, Random Forest ,Decision tree. Out of which Random Forest and Logistic Regression  model’s performance were outstanding. Able to predict stroke with 95% accuracy by using Logistic Regression and Random Forest.

Technology:
Python/Machine learning

Reference:
Kaggle
  




