import json
import plotly
import pandas as pd
import numpy as np

from flask import Flask
from flask import render_template, request, jsonify
import joblib
from wtforms import SelectField
from flask_wtf import FlaskForm
import os

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

clf_rf = joblib.load("../models/clf_rf.pkl")
clf_nb = joblib.load("../models/clf_nb.pkl")

class Form(FlaskForm):    
    
    # options for gender in the dropdown menu
    gender_list = [('gender_F','Female'), 
                   ('gender_M','Male'), 
                   ('gender_O', 'Other'),
                   ('gender_U', 'Unknown')]
    gender = SelectField('gender', choices=gender_list)
    
    # options for age in the dropdown menu
    age_list = [('age_group_<20', '<20'),
                ('age_group_20-25', '20-25'),
                ('age_group_25-30', '25-30'),
                ('age_group_30-35', '30-35'),
                ('age_group_35-40', '35-40'),
                ('age_group_40-45', '40-45'),
                ('age_group_45-50', '45-50'),
                ('age_group_50-55', '50-55'),
                ('age_group_55-60', '55-60'),
                ('age_group_60-65', '60-65'),
                ('age_group_65-70', '65-70'),
                ('age_group_70-80', '70-80'),
                ('age_group_80-101', '80-101'),
                ('age_group_unclear','unclear')]    
    age = SelectField('age', choices=age_list)
    
    # options for membership history in the dropdown menu
    membership_list = [('membership_group_before_2014', 'before 2014'),
                       ('membership_group_since_2014', 'since 2014'),
                       ('membership_group_since_2015', 'since 2015'),
                       ('membership_group_since_2016', 'since 2016'),
                       ('membership_group_since_2017', 'since 2017'),
                       ('membership_group_since_2018', 'since 2018')]   
    membership = SelectField('become member since when', choices=membership_list)
    
    # options for income in the dropdown menu
    income_list = [('income_group_<40k', '<40k per year'),
                   ('income_group_40-60k', '40-60k per year'),
                   ('income_group_60-80k', '60-80k per year'),
                   ('income_group_80-100k', '80-100k per year'),
                   ('income_group_>100k', '>100k per year'),
                   ('income_group_unknown','income unknown')]
    income = SelectField('income', choices=income_list)
                           

# index webpage displays cool visuals and receives user input text for model
@app.route('/', methods=['GET','POST'])

def index():
    
    form = Form()
    
    # get the input data for gender, age, membership history and income for the customer
    gender = form.gender.data
    age = form.age.data
    membership = form.membership.data
    income = form.income.data
    
    # create a dataframe with feature columns below as the input for the machine learning model
    columns = ['gender_F', 'gender_M', 'gender_O', 'gender_U', 'age_group_20-25',
       'age_group_25-30', 'age_group_30-35', 'age_group_35-40',
       'age_group_40-45', 'age_group_45-50', 'age_group_50-55',
       'age_group_55-60', 'age_group_60-65', 'age_group_65-70',
       'age_group_70-80', 'age_group_80-101', 'age_group_<20',
       'age_group_unclear', 'membership_group_before_2014',
       'membership_group_since_2014', 'membership_group_since_2015',
       'membership_group_since_2016', 'membership_group_since_2017',
       'membership_group_since_2018', 'income_group_40-60k',
       'income_group_60-80k', 'income_group_80-100k', 'income_group_<40k',
       'income_group_>100k', 'income_group_unknown']
    
    X = pd.DataFrame(np.zeros((1,len(columns))), columns=columns)
    
    # fill in the input data
    if gender in columns:
        X.loc[0,[gender,age,membership,income]]=1
    
    # initialize an empty numpy array for output
    Y_pred = np.zeros((1,len(clf_nb.estimators_)))
    
    # use random forest classifier for predicting offer 4 
    for i,clf in enumerate(clf_rf.estimators_):
        if i == 4:
            Y_pred[:,4] = clf.predict(X)

    # use naive bayes classifier for all the rest offer types
    for i,clf in enumerate(clf_nb.estimators_):
        if i != 4:
            Y_pred[:,i] = clf.predict(X)
   
    # a list of offer info to be shown in the output table
    offer_description = ['Offer 0: bogo (10/10), 7days via email,mobile,social',
                         'Offer 1: bogo (10/10), 5days via web,email,mobile,social',
                         'Offer 2: informational, 4days via web,email,mobile',
                         'Offer 3: bogo (5/5), 7days via web,email,mobile',
                         'Offer 4: discount (20/5), 10days via web,email',
                         'Offer 5: discount (7/3), 7days via web,email,mobile,social',
                         'Offer 6: discount (10/2), 10days via web,email,mobile,social',
                         'Offer 7: informational, 3days via email,mobile,social',
                         'Offer 8: bogo (5/5), 5days via web,email,mobile,social',
                         'Offer 9: discount (10/2), 7days via web,email,mobile']
    
    # render web page with plotly graphs
    return render_template('master.html', form=form, offer_description=offer_description, Y_pred=Y_pred)



def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
