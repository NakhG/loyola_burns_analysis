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
os.chdir("C:\\Users\\gnakhleh\\Documents\\Loyola")

ca_burns_06to11 = pd.read_stata("Burns Data\\Burns CA 2006 thru 2011.dta")

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

burns_died = burns_slice[burns_slice['DIED'] == 1]['AGE']
burns_died.head()
burns_lived = burns_slice[burns_slice['DIED'] == 0]['AGE']
burns_lived.head()

#Make the histograms
sns.kdeplot(burns_died, shade=True)
sns.kdeplot(burns_lived, shade=True).axes.set_title('Distributions (Burn victims only)', fontsize=20,color="black");