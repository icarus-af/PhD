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

n = ESD_Test(red, 0.05, 5)

# newdata = remove_outliers(data, n)

Gcal, x = grubbs_stat(dE)

Gcrit = calculate_critical_value(len(dE), 0.05)

#%% CHOOSE COLORS
reps = 10

if reps < 7:

    colors = ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF']

else:
    
    colors = plt.get_cmap('rainbow')(np.linspace(0, 1, reps))

# colors=['#FF5252']*60
#%% RUN LOOP (tech, sep, reps, scan, b, Emax, Emin)

peaks('cv', '	', reps, 2, 'f', .9, -.4)

#%% RUN INDIVIDUAL (item, sep, tech, scan, b, Emax, Emin)

plot(2, '	', 'cv', 2, 'f', 0, -.2)
plot(3, '	', 'cv', 2, 'f', 0, -.2)

#%%

peaks = np.array([plot(item) for item in [0,1,4]])
