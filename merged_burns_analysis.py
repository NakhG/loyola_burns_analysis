# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:34:25 2016

@author: gnakhleh
"""

'''
Loyola BURNS analysis
'''

import numpy as np
import pandas as pd

import os
os.chdir("C:\\Users\\gnakhleh\\Documents\\Loyola\\Burns Data")


#Read in merged data

merged_data = pd.read_csv("commo_burns.csv")

merged_data.shape #(78773, 36)

merged_data.columns

#Remove na's in Age column
pd.isnull(merged_data.AGE).value_counts()  #there are none


#Look at the data, do we need to pre-process?
merged_data.head(11)

#Rename "less than 18" to "17 & under"
merged_data.Age_ranges[merged_data.Age_ranges == 'Less than 18'] = '17 & under'


#We only want people where DX_BURN = 1
burns_data = merged_data[merged_data.DX_BURN == 1]
burns_data.head(11)
burns_data.shape #(33000, 36)

#Some EDA: mortality rate by age group
crosstab_agerange_died = pd.crosstab(burns_data.Age_ranges, burns_data.DIED)
crosstab_agerange_died_percent = crosstab_agerange_died.apply(lambda x: x/x.sum(), 1)

#graph it

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(8, 5))
agerange_mortality_plot = sns.barplot(x=crosstab_agerange_died_percent.index, y=crosstab_agerange_died_percent[1], color='maroon')
agerange_mortality_plot.axes.set(title="Mortality rate by age range (burn victims only)", xlabel='Age ranges', ylabel='mortality rate')

#Repeat, but for ages, not age ranges
crosstab_age_died = pd.crosstab(burns_data.AGE, burns_data.DIED)
crosstab_age_died_percent = crosstab_age_died.apply(lambda x: x/x.sum(), 1)

crosstab_age_died_percent[1].plot()  #NOTE: not meaningful due to sample sizes

#Begin modeling a logistic regression to predict death from burns
#For starters, will look like: DIED ~ Age + (all the CM_)

from sklearn.linear_model import LogisticRegression

#Need to make an X and y: predictors and target
burns_data_noNA = burns_data.dropna(axis=0)
y = burns_data_noNA.DIED

col_names = list(burns_data_noNA.columns.values)
predictors_list = [s for s in col_names if 'cm_' in s.lower()]
predictors_list.append('AGE')

X = burns_data_noNA[predictors_list]
X.shape

#Make training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 191, stratify = y)

X_train.shape
X_test.shape
y_train.shape


model = LogisticRegression()
model.fit(X_train, y_train)
print(model)

model.score(X_test, y_test) #97% accurate, seems odd

#Investigate the model
#Likely so accurate because most patients live
100 * burns_data.DIED.value_counts() / len(burns_data.DIED)
#yep, exactly 97%. The real test is sensitivity

predicted = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(predicted) ; print(probabilities) #bad sign: maybe it labeled everything 0

from sklearn import metrics
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probabilities[:, 1]))

#ok fine, but tough to interpret
#confusion matrix would be easier

print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

#That confirms it: poor performance on whether they die



