# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:52:07 2023

@author: lesql
"""

# -*- coding: utf-8 -*-


import sys
sys.path.insert(0, 'G:\My Drive\Atual (1)\PESQUISA\GitRepos\PhD')
sys.path.insert(0, 'G:\Meu Drive\Atual (1)\PESQUISA\GitRepos\PhD')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_treatment_model_temp as dt

#%%


df = pd.read_csv('alldata-3-ALL.csv')

#%%
data = df


x1 = data['alt']
x2 = data['sep']
x3 = data['vel']
x4 = data['pot']

X = [x1, x2, x3, x4]
Y = data['dE']

# err = data['std']
# err = np.array([df['dE}_std'].min()]*17)


err = 0

n = 4
m = 3

# select = [0, 3, 6, 7]

model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled = dt.main_regmodel(n, m, 'WLS', X, Y, test_all=False)

#%%

df2=df.groupby(np.arange(len(df))//3).mean()
df3=df.groupby(np.arange(len(df))//3).std(ddof=1)


dps = [ [df2['alt'][i],df2['sep'][i],df2['vel'][i],df2['pot'][i]] for i in range(17)] 

plt.errorbar(np.linspace(0,1,17), df2['dE'], color=black, yerr=df3['dE'], fmt='o')

plt.ylabel('dE')
plt.xlabel('[alt, sep, vel, pot]')
plt.xticks([])

#%%

