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
os.chdir("C:\\Users\\gnakhleh\\Downloads")

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

#Make overlapping histograms: x = age, y = % died. One for DX_BURN = 1, One for = 0

from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

#I want every row where DX_BURN == 1, and the columns 'KEY', 'AGE', 'DIED', 'DX_BURN'

burns_slice = ca_burns_06to11[ca_burns_06to11['DX_BURN']==1]
burns_slice = ca_burns_06to11[['KEY', 'AGE', 'DIED', 'DX_BURN']]

age_burns_died = burns_slice[burns_slice['DIED'] == 1]['AGE']
age_burns_died.head()
age_burns_lived = burns_slice[burns_slice['DIED'] == 0]['AGE']
age_burns_lived.head()

#Make the histograms
sns.kdeplot(age_burns_died, shade=True)
sns.kdeplot(age_burns_lived, shade=True).axes.set_title('Distributions (Burn victims only)', fontsize=20,color="black");
#age dist. of burn victims who died is skewed more to the right

#Is that statistically meaningful?
#Could do a t-test (answering: is there a significatn diff. in the means of the ages of burn patients who lived vs those who died?)
#from scipy.stats

np.random.seed(340)
stats.ttest_ind(np.array(age_burns_died), np.array(age_burns_lived))

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

burns_slice[burns_slice['AGE'] > 100][['AGE','DIED']]


