# udacity_project_starbucks
 
Udacity Data Science Nanodegree capstone project - Starbucks marketing problem


## Installation
Several libraries are used in the code file and should be pip install first (if not already):
- numpy
- pandas 
- sklearn
pickle
json
plotly
flask 
joblib
wtforms 
flask_wtf 
os


## Instruction
Input data are included in the data folder:
portfolio.json - offer description

profile.json - customer(user) information including gender,age,membership history and income

transcript.json - customer(user) transaction and offer interaction records

To see the model training process (including data import/analysis/cleaning) and get the model pickle file:
- go to the "Starbucks_Capstone.ipynb" file

To run the web app for offer recommendation:
- go to the "app" folder and run python run.py in command line
- go to http://0.0.0.0:3001/


## Overview
In this project, the raw data about Starbucks customer, a list of product offers, and offer-customer interactions are analyzed, cleaned, and used to train a machine learning model for predicting which offer to send for each customer. Four user features are used in the model: (1) gender group, (2) age group, (3) how long has the user been a member, (4) income group. Dummy variables are created for each of the feature for model training.

After data cleaning and visualization, an important step is to define user/customer's response to offers. Here I define:

A positive response (1) = should send this offer to this user.

A negative response (0) = no need to send this offer to this user.

An unclear response (NAN) = unclear because the user doesn't view the offer

**For informational-type offers (2, 7)**:
The response is considered positive if the user views the offer and then makes a purchase within the offer duration (96hr for offer 2, 72hr for offer 7).
The response is negative if the user views the offer but does not make any purchase within the duration.
If the user does not view the offer, the response is labrled "NAN" (unclear) given that the user is not even aware of the offer.

**For other offer (0,1,3,4,5,8,9)**:
The response is considered positive if the user views the offer and completes the offer "task".
However, if there is any record of completing the "task" without viewing, then the response is considered negative. This is because the user will probably purchase the product even without the offer in this case.
If the user views the offer but does not complete the task, it is also considered as negative response.
If the user doesn't view the offer nor complete the task, it is considered as unclear response.

With the model input/output defined, I further use FunkSVD to extrapolate unknown use/offer interactions. This provides more training data for each offer type and is found to improve the model performance on test set. Note that the test set performance is only evaluated with known user-offer interaction (but NOT on estimated responses from FunkSVD).


## Conclusion/Summary
In this work, I took a modeling approach to achieve the goal of predicting who should receive each type of offers. I frist create a user response dataframe by defining what "positive response" means for each type of offer. Then I define my user features as age group, membership history group, gender group, income group and train a few classfiers on my training set data. I also used FunkSVD method to fill in all the missing data for the user-offer interaction matrix. Note that this is just for improving model training and does not affect the integrity and validity of model evaluation, as it is not done for the test set data.

Multinominal NB classifier is found to perform well for most offer types, but Random Forest classifier outperforms it for offer type 4. Therefore, a combined-model classifier is used to make predictions for all offer types. Combining the two classifiers, the test set F1 score is:

0.8~0.84 for offer 2,5,6
0.7~0.8 for offer 0,1,7,8
0.64 for offer 9
0.59 for offer 3
0.48 for offer 4

For any user in our database with features(age, member_since, gender, income) available, we can simply decide which offer to send based on the model prediction (1-send, 0-do not send)

Based on this model, I build a web application to allow an input of user information and provide a recommendation on what offers to send for this user.
