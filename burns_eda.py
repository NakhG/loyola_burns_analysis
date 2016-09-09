# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:09:51 2016

@author: gnakhleh
"""

'''
Loyola Research

Analyzing the burns datasets
Question: at what age do burns seem to be more fatal?

Files are in State format
'''

import numpy as np
import pandas as pd

import os
os.chdir("C:\\Users\\gnakhleh\\Documents\\Loyola\\Burns Data")

ca_burns_06to11 = pd.read_stata("Burns CA 2006 thru 2011.dta")

ca_burns_06to11.shape #36,464 rows & 223 columns

#create a sample to inspect by hand in Excel
sample_ca_burns_06to100 = ca_burns_06to11.sample(150, random_state=120)

sample_ca_burns_06to100.to_csv("Burns Data\\Sample Burns CA 2006 thru 2011.csv")

'''
ok looking at the data, we see a few columns of interest
DX_BURN: a burn diagnosis? 1/0
Age: in years
Died: 1/0
'''

#Let's do a crosstab of DX_Burn and Died
pd.crosstab(ca_burns_06to11.AGE, [ca_burns_06to11.DX_BURN, ca_burns_06to11.DIED])
#that's likely valuable, but a bit tough to interpret

#Make overlapping histograms: only those with burns. age dist. of those who lived vs died

from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

#I want every row where DX_BURN == 1, and the columns 'KEY', 'AGE', 'DIED', 'DX_BURN'
burns_slice = ca_burns_06to11[ca_burns_06to11.DX_BURN == 1]
burns_slice = burns_slice[['KEY', 'AGE', 'DIED', 'DX_BURN']]

age_burns_died = burns_slice[burns_slice['DIED'] == 1]['AGE']
age_burns_died.head()
age_burns_lived = burns_slice[burns_slice['DIED'] == 0]['AGE']
age_burns_lived.head()

#Make the histograms
#This shows age ranges of those who live from burns (green) and those who die from burns (blue)
sns.kdeplot(age_burns_died, shade=True)
sns.kdeplot(age_burns_lived, shade=True).axes.set_title('Distributions (Burn victims only)', fontsize=20,color="black");
#age dist. of burn victims who died is skewed more to the right

#Is that statistically meaningful?
#Could do a t-test (answering: is there a significatn diff. in the means of the ages of burn patients who lived vs those who died?)
#from scipy.stats

np.random.seed(340)
stats.ttest_ind(np.array(age_burns_died), np.array(age_burns_lived))
#that doesn't work due to NA's

#is the difference meaningful?
stats.ttest_ind(np.array(age_burns_died.dropna()), np.array(age_burns_lived.dropna()))


#ok so it looks like the mean age of burn victims differs meaningfully by age
#now what?
#if we want to get an age range that is helpful, we are looking for cutoffs
#when does the risk rise higher?

#we'd want the fatality rate by age

burns_agegrouping = burns_slice[['AGE','DIED']].dropna().sort('AGE').groupby('AGE')
burns_agegrouping.head()
burns_agegrouping.aggregate('count')
burns_deathrates_proto = burns_agegrouping.aggregate([np.sum, 'count'])
burns_deathrates_proto.to_csv("deathrates.csv")

#at this point, we don't want to rely on cleaning up this grouped view in Excel
#have to do the work in Python for replicability
#We'd want to see death rate by age
#Probably should ignore outlier ages, too. They make it hard to see the smaller % changes

#Let's make age ranges
age_bins = {'18-25':list(range(18,26)), '26-33':list(range(26,34)),'34-41':list(range(34,42)), '42-49':list(range(42,50)), '50-57':list(range(50,58)), '58-65':list(range(58,66)), '66-73':list(range(66,74)), '74-81':list(range(74,82)), '82-89':list(range(82,90)), '90+':list(range(90,int(ca_burns_06to11.AGE.max())))}
ca_burns_06to11.age_bins = 0

#want to go through every row and assign value to age_bins based on AGE value

def age_range_finder(x):
    for key, value in age_bins.items():
        if x in value:
            return key

ca_burns_06to11['age_bins'] = ca_burns_06to11['AGE'].apply(age_range_finder)

ca_burns_06to11[['AGE', 'age_bins']].head(8)

#plot age range frequencies
sns.countplot(x='age_bins', data=ca_burns_06to11.sort('age_bins'), color="salmon")
#age ranges have a normal distribution, a bit of rightward skew

#now, we want to analyze death rates by age_group
burns_slice = ca_burns_06to11[ca_burns_06to11.DX_BURN == 1]

#visualize the burn victims as a subset of overall data
f, ax = plt.subplots(figsize=(6, 5))
sns.countplot(x='age_bins', data=ca_burns_06to11.sort('age_bins'), color="salmon", label="Overall")
sns.countplot(x='age_bins', data=burns_slice.sort('age_bins'), color="red", label="DX_BURN = 1")
ax.legend(ncol=2, loc="upper right", frameon=True)
ax.set(title="Age range counts (CA only)", xlabel="Age ranges")

burns_slice = burns_slice[['KEY', 'AGE', 'age_bins', 'DIED', 'DX_BURN']]
burns_agerangegrouping = burns_slice[['AGE','age_bins','DIED']].dropna().sort('AGE').groupby('age_bins')
burns_deathrates_proto1 = burns_agerangegrouping.aggregate([np.sum, 'count'])
burns_deathrates_proto1['DIED']

#create death rate variable
burns_deathrates_proto1['death_rate'] = burns_deathrates_proto1['DIED']['sum'] / burns_deathrates_proto1['DIED']['count']

#plot
f, ax = plt.subplots(figsize=(6, 5))
agerange_mortality_plot = sns.barplot(x=burns_deathrates_proto1.index, y=burns_deathrates_proto1['death_rate'], color='maroon')
agerange_mortality_plot.axes.set(title="Mortality rate by age range (CA only, burn victims only)", xlabel='Age ranges', ylabel='mortality rate')

#do ttest on deathrate by age range
stats.ttest(burns_deathrates_proto1[['death_rate']])


#or instead
burns_deathrates_proto2 = pd.crosstab(burns_slice.age_bins, burns_slice.DIED)
burns_deathrates_proto2['Total'] = burns_deathrates_proto2[0.0] + burns_deathrates_proto2[1.0]


#plot
sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(6, 5))
sns.set_color_codes("pastel")
sns.barplot(x=burns_deathrates_proto2['Total'], y=burns_deathrates_proto2.index, 
            data=burns_deathrates_proto2, label="Total", color="b")
sns.set_color_codes("muted")
sns.barplot(x=burns_deathrates_proto2[1.0], y=burns_deathrates_proto2.index, 
            data=burns_deathrates_proto2, label="Died", color="b")
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlabel='# Patients')

#what now?
#HOW do we answer the main question (coming up w/ an age range/threshold where burns are more deadly), STATISTICALLY:
# correlation of age with fatality for burn victims (vs non burn victims?)
# regression (how to pick from all variables? PCA? Factor analysis?)
# ^^^ for both of those: need to control for comorbidities (try a correlation matrix)
# "U-test": nonparametric for ages
# T-test for age ranges


