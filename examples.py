# -*- coding: utf-8 -*-

from patsy import dmatrices

import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn import preprocessing

import pandas as pd
import numpy as np

import dash
from dash import html, dash_table
from dash import dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from uncertainties import ufloat
from uncertainties.umath import *

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

from scipy.interpolate import UnivariateSpline
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
import matplotlib as mpl


# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Times New Roman']
# rcParams['font.sans-serif'] = ['Lato']
rcParams['axes.labelpad'] = 15
plt.rcParams['font.size'] = 15
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["font.weight"] = "bold"

colors = ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF' , '#FF5252', '#FF5252', '#FF5252', '#FF5252', '#FF5252', '#FF5252', '#FF5252', '#FF5252']

## 6 cores ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF']

## 6 cores ['preto', 'vermelho', 'amarelo', 'verde', 'azul', 'rosa']

#%% For ESDtest or Grubbs

data = pd.read_csv('testeariel.txt', sep=' ')

ox = np.array(data['ox'])
red = np.array(data['red'])
dE = np.array(data['dE'])
ioveri = ox/-red

my_dict = {'ox': ox/max(ox), 'red': red/min(red), 'dE': dE/max(dE), 'ioveri': ioveri/max(ioveri)}

bxplt = boxplots(my_dict)

#for Grubbs test n = 1
#for ESD         n > 1

n = ESD_Test(red, 0.05, n)

# newdata = remove_outliers(data, n)

Gcal, x = grubbs_stat(dE)

Gcrit = calculate_critical_value(len(dE), 0.05)

#%% RUN LOOP (tech, sep, reps, scan, b, Emax, Emin)

peaks('cv', '	', reps, 2, 'f', .9, -.4)

#%% RUN INDIVIDUAL (item, sep, tech, scan, b, Emax, Emin)

plot(2, '	', 'cv', 2, 'f', 0, -.2)
plot(3, '	', 'cv', 2, 'f', 0, -.2)

#%% RUN LOOP

peaks = np.array([plot(item) for item in [0,1,4]])

#%% For data_treatment_model

# Read your data
data = df

# State your variables, responses and errors

x1 = data['freq']
x2 = data['amp']
x3 = data['step']

y1 = data['I']

# err = data['std']
# err = np.array([df['dE_std'].min()]*17)
# y1 = y1/y1.max()

# Scale and finalize X matrix

X_final = pd.DataFrame([x1, x2]).T

X_final_scaled, X_poly_final_scaled, scaler = poly_scale(X_final, 3)

# Select final variables

select = [0,2,3,4,5,6,7,8,9,10,12,13,14,16,17,18,19]

# Calculate model and get parameters

model_final, coefs_sm, r2, r2_adj = reg_model( y1, X_poly_final_scaled[:,select], 'OLS', err)

print(model_final.summary())

diff = allcoefs_2.difference(select)

coefs = np.zeros(35)

for i in diff:
    coefs[i] = 0
for i in np.arange(0, len(select)):
    coefs[select[i]] = coefs[i]
    
effects = np.abs(coefs_sm)/(np.sum(np.abs(coefs_sm))-coefs_sm[0])*100

print('')
print("="*78)
print('effects = ', effects)
print("="*78)
print('')

