# Marketing_Campaign_Prediction
This repository contains Python code for a marketing campaign prediction project. The goal of this project is to predict the success of marketing campaigns based on various features such as contact type, response, housing loan status, and campaign details.
# Project Structure
The project is structured as follows:

Data: The project uses a dataset located at C:\Users\ASUS\Downloads\marketing.csv. The test data for predictions is sourced from C:\Users\ASUS\Downloads\marketing_test.xlsx.

Code: The main code file is named marketing_campaign_prediction.py. This file contains Python code to read the dataset, preprocess the data, train machine learning models (Decision Tree Classifier and Random Forest Classifier), optimize the model, and make predictions.
# Data Preprocessing
The initial dataset is loaded using pandas and basic exploratory data analysis is performed.
Irrelevant columns such as 'job' and 'month' are dropped.
Missing values are handled appropriately.
Categorical variables are converted to numerical using one-hot encoding.
# Model Training and Optimization
The dataset is split into training and testing sets.
Two classifiers, Decision Tree Classifier and Random Forest Classifier, are trained on the data.
The Random Forest Classifier is further optimized using Randomized Search Cross-Validation.
Feature importance is assessed and less relevant features are removed for model optimization.

# Model Evaluation
Model accuracy is evaluated using various metrics including accuracy score, confusion matrix, ROC-AUC score, and Gini score.
ROC curves are plotted to visualize model performance.
# Making Predictions
The optimized model is used to make predictions on new data.
The test data is loaded from C:\Users\ASUS\Downloads\marketing_test.xlsx and preprocessed in a manner consistent with the training data.
Predictions are made and stored in the 'Prediction' column of the test dataset.

# Result Interpretation
The final predictions can be found in the 'Prediction' column of the test_data DataFrame. These predictions represent the probability of success for the marketing campaigns.
