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

from tqdm import tqdm

#%%



df = pd.read_csv('0915_DOE_redo.csv')


data = df


x1 = data['freq']
x2 = data['amp']
x3 = data['step']
# x4 = data['pot']

X = [x1, x2, x3]
Y = data['I']

# errors = data['dE_std']
# err = np.array([df['dE}_std'].min()]*17)


err = 0

n = 3
m = 3

select=[0,1,2,6,7,8,12,16,18]

model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled = dt.main_regmodel(n, m, 'OLS', X, Y, test_all=False, select=select)
#%%
dt.main_surface(n, m, coefs, X, Y, fixposition=0, fixvalue=0)

#%%

df = pd.read_csv('alldata-3.csv')

#%%
data = df


x1 = data['alt']
x2 = data['sep']
x3 = data['vel']
x4 = data['pot']

X = [x1, x2, x3, x4]
Y = data['dE']

errors = data['dE_std']
# err = np.array([df['dE}_std'].min()]*17)


err = 0

n = 4
m = 3

select = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

select=[0,8,9,11,13,14,18,20,21,23,24,25,27,29,30,32,33]
select = [0,9,11,13,20,23,25,27,29,30]
select = [0,13,20,23,25,27,29,30]
# select = [13,25,29]
# select = [13,25,29]

model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled = dt.main_regmodel(n, m, 'WLS', X, Y, err=errors, select=select)

#%%

dt.main_surface(4,3,coefs,X,Y, fixposition=[1,3], fixvalue=1)


#%%


coefs_str =  ['intercept' ,  'x1'  , 'x2'  ,  'x3'  ,  'x4'  ,  'x1*x1'  ,  'x1*x2'  ,  'x1*x3'  ,  'x1*x4' , 'x2*x2'  , 'x2*x3'  ,  'x2*x4'  ,  'x3*x3'  ,  'x3*x4'  , 'x4*x4'  , 'x1*x1*x1'  ,  'x1*x1*x2'  ,  'x1*x1*x3'  ,  'x1*x1*x4'  ,  'x1*x2*x2'  , 'x1*x2*x3'  ,  'x1*x2*x4'  , 'x1*x3*x3'  , 'x1*x3*x4'  , 'x1*x4*x4'  , 'x2*x2*x2'  ,  'x2*x2*x3'  ,  'x2*x2*x4'  , 'x2*x3*x3'  , 'x2*x3*x4'  ,  'x2*x4*x4'  , 'x3*x3*x3'  ,  'x3*x3*x4'  ,  'x3*x4*x4'  , 'x4*x4*x4']

coefs_str_ = pd.DataFrame(coefs_str)[coefs != 0]

df2 = coefs_str_.replace('x1','alt', regex=True)
df2 = df2.replace('x2','sep', regex=True)
df2 = df2.replace('x3','vel', regex=True)
df2 = df2.replace('x4','pot', regex=True)


#%%
for i in range(len(coefs_str_)):
    print(coefs[coefs != 0][i], '*', np.array(coefs_str_).ravel()[i], ' + ') 
    
#%%

df = pd.read_csv('alldata-3.csv')

train = df.sample(n=12)
test = df.drop(train.index)

#%%

data = train


x1 = data['alt']
x2 = data['sep']
x3 = data['vel']
x4 = data['pot']

X = [x1, x2, x3, x4]
Y = data['dE']

errors = data['dE_std']
# err = np.array([df['dE}_std'].min()]*17)


err = 0

n = 4
m = 3

select = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

select=[0,8,9,11,13,14,18,20,21,23,24,25,27,29,30,32,33]
select = [0,9,11,13,20,23,25,27,29,30]
select = [0,13,20,23,25,27,29]
# select = [13,25,29]
# select = [13,25,29]

model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled = dt.main_regmodel(n, m, 'WLS', X, Y, err=errors, select=select)
dt.main_surface(4,3,coefs,X,Y, fixposition=[1,3], fixvalue=1)
