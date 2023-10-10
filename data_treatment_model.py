# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:20:51 2023

@author: lesql
"""

import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np

import dash
from dash import html, dash_table
from dash import dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from uncertainties import ufloat
from uncertainties.umath import *

from itertools import combinations

from tqdm import tqdm

rcParams['axes.labelpad'] = 15
plt.rcParams['font.size'] = 12

## 6 cores ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF']

# 6 cores ['preto', 'vermelho', 'amarelo', 'verde', 'azul', 'rosa']

black, red, yellow, green, blue, pink = '#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF'
#%% Funções

def get_summaries(model):
    
    table1 = model.summary().tables[0].as_html()
    sumarry_table1 = pd.read_html(table1, header=0, index_col=0)[0].reset_index()
    
    table2 = model.summary().tables[1].as_html()
    sumarry_table2 = pd.read_html(table2, header=0, index_col=0)[0].reset_index()
    
    table3 = model.summary().tables[2].as_html()
    sumarry_table3 = pd.read_html(table3, header=0, index_col=0)[0].reset_index()
    
    return sumarry_table1, sumarry_table2, sumarry_table3


    
def add_Scatter(fig, df, color, name):
    
    fig.add_trace(go.Scatter(x=df['Potential applied (V)'], y=df['WE(1).Current (A)']*1e3, line=dict(color=color, width=2), name=name
        
        ))
    
    return

def scale(df):
    
    scaler = MinMaxScaler()
    
    df_scaled = scaler.fit_transform(df)
    
    return df_scaled

def poly_scale(df, degree):
    
    poly = PolynomialFeatures(degree=degree)
    scaler = MinMaxScaler()
    
    df_scaled = scaler.fit_transform(df)
    
    df_poly_scaled = poly.fit_transform(df_scaled)
    
    return df_scaled, df_poly_scaled, scaler

def reg_model(y, x, method='OLS', err=None):
    """
    Fits and report a linear model ({method} = 'OLS' or 'WLS') for y(x) returning (model, coefs, r2, r2_adj)
    """
    if method == 'OLS':
    
        model = sm.OLS(y, x).fit()
        
    elif method == 'WLS':
        
        weights =  1/(err**2)
        model = sm.WLS(y, x, weights=weights).fit()
        
    coefs = np.array(model.params)    
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    
    print(model.summary())
    
    return model, coefs, r2, r2_adj


def make_surface3D(x1, x2, x3, coefs, fun):
    
    surf = np.array(fun(np.ravel(x1), np.ravel(x2), np.ravel(x3), coefs))
    surface = surf.reshape((100,100))
    
    return surface

def make_surface4D(x1, x2, x3, x4, coefs, fun):
    
    surf = np.array(fun(np.ravel(x1), np.ravel(x2), np.ravel(x3), np.ravel(x4), coefs))
    surface = surf.reshape((100,100))
    
    return surface

def make_1D(min_x, max_x, step):
    
    x = np.linspace(min_x, max_x, step)
    y = np.linspace(min_y, max_y, step)
    X, Y = np.meshgrid(x, y)
    
    return X, Y

def make_2D(min_x, max_x, min_y, max_y, step):
    
    x = np.linspace(min_x, max_x, step)
    y = np.linspace(min_y, max_y, step)
    X, Y = np.meshgrid(x, y)
    
    return X, Y

def get_max(surface, X, Y, scaler):

    value = np.where(surface == np.amax(surface))
    
    position = [X[value[0],value[1]], X[value[0],value[1]]]

    position_df = pd.DataFrame(position, [1,2])
    position_df['x'] = [Y[value[0],value[1]],Y[value[0],value[1]]]
    pos_unscaled = scaler.inverse_transform(position_df) 
    
    return pos_unscaled[0]
    

#%% Funções Específicas

def add_Scatter(fig, df, color, name):
    
    trace = fig.add_trace(go.Scatter(x=df['Potential applied (V)'], y=df['WE(1).Current (A)']*1e3, line=dict(color=color, width=2), name=name
        
        ))
    
    return trace

def add_Fig(df, color, name):
    
    fig = go.Figure(data=[go.Scatter(x=df['WE(1).Potential (V)'], y=df['WE(1).Current (A)']*1e3, line=dict(color=color, width=2), name=name
        
        )
        
        ])
    
    return fig

def up_layout(fig, title, x, y):
    
    fig.update_layout(font_family='Lato', font_color='black', font_size=15,
                              
                            
        title=title,title_x = 0.5,        
        yaxis_title=y,
        xaxis_title=x,
        #plot_bgcolor='white'
        autosize=True,
        #margin=go.Margin(l=0, r=0, t=0, b=0),
        #paper_bgcolor='#242424',
        legend=dict(
            x=0.005,
            y=.95,
            #traceorder="normal",
            font=dict(
                family="Lato",
                size=15,
                color="black"
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        #yaxis_range=[-.5, 0.5],
        #xaxis_range=[-.3, 1.1]
        
        )
    
    return
#%%

def score_graph(df, err, title):
    
    x = np.arange(0, 33, 1)

    fig = go.Figure(data=[go.Bar(name='', x=x ,y = df, marker_color='#343E3D', error_y=dict(type='data', array=err, color='#FF5252') )
                            
                            ])


    fig.update_layout(font_family='Lato', font_color='black', font_size=15,
                              
                            
        title=title,title_x = 0.5,        
        yaxis_title=title,
        xaxis_title="Experiments",
        autosize=True,
        
        legend=dict(
            x=0.005,
            y=.95,
            font=dict(
                family="Lato",
                size=15,
                color="black"
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        
        )

    fig.show()
    
    return fig

#%%


def get_allcoefs(n, m):
    
    if n == 1:
        
        if m == 1:
            
            allcoefs = {0,  1}
 
        elif m == 2:
            
            allcoefs = {0,  1,  2}
            
        elif m == 3:
            
            allcoefs = {0,  1,  2,  3}
            
    if n == 2:
        
        if m == 1:
            
            allcoefs = {0,  1,  2}
            
        elif m == 2:
            
            allcoefs = {0,  1,  2,  3,  4,  5}
            
        elif m == 3:
            
            allcoefs = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9}
            
    if n == 3:
        
        if m == 1:
            
            allcoefs = {0,  1,  2, 3}
            
        elif m == 2:
            
            allcoefs = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9}
            
        elif m == 3:
            
            allcoefs = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
            
    if n == 4:
        
        if m == 1:
            
            allcoefs = {0,  1,  2, 3, 4}
            
        elif m == 2:
            
            allcoefs = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14}
            
        elif m == 3:
            
            allcoefs = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}      
            
        elif m == 4:
            
            allcoefs = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69}  
        
    return allcoefs     


#%%
def allfun(n, m, X, coefs):

    if n == 1:
        
        if m == 1:
        
            def fun(X, coefs):
                x1 = X
                return coefs[0] + coefs[1]*x1
            
        elif m == 2:
            
            def fun(X, coefs):
                x1 = X
                return coefs[0] + coefs[1]*x1 + coefs[2]*x1*x1
            
        elif m == 3:
            
            def fun(X, coefs):
                x1 = X
                return coefs[0] + coefs[1]*x1 + coefs[2]*x1*x1 + coefs[3]*x1*x1*x1
            
    elif n == 2:

        if m == 1:
        
            def fun(X, coefs):
                x1 = X[0]
                x2 = X[1]
                return coefs[0] + coefs[1]*x1 + coefs[2]*x2
            
        elif m == 2:
            
            def fun(X, coefs):
                x1 = X[0]
                x2 = X[1]
                return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x1*x1 + coefs[4]*x1*x2 + coefs[5]*x2*x2
            
        elif m == 3:
            
            def fun(X, coefs):
                
                x1 = X[0]
                x2 = X[1]

                return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x1*x1 + coefs[4]*x1*x2 + coefs[5]*x2*x2 + coefs[6]*x1*x1*x1 + coefs[7]*x1*x1*x2 + coefs[8]*x1*x2*x2 + coefs[9]*x2*x2*x2
            
    elif n == 3:

        if m == 1:
        
            def fun(X, coefs):
                x1 = X[0]
                x2 = X[1]
                x3 = X[2]
                return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3
            
        elif m == 2: 
            
            def fun(X, coefs):
                x1 = X[0]
                x2 = X[1]
                x3 = X[2]
                return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3 + coefs[4]*x1*x1 + coefs[5]*x1*x2 + coefs[6]*x1*x3 + coefs[7]*x2*x2 + coefs[8]*x2*x3 + coefs[9]*x3*x3
            
        elif m == 3:
            
            def fun(X, coefs):
                x1 = X[0]
                x2 = X[1]
                x3 = X[2]
                return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3 + coefs[4]*x1*x1 + coefs[5]*x1*x2 + coefs[6]*x1*x3 + coefs[7]*x2*x2 + coefs[8]*x2*x3 + coefs[9]*x3*x3 + coefs[10]*x1*x1*x1 + coefs[11]*x1*x1*x2 + coefs[12]*x1*x1*x3 + coefs[13]*x1*x2*x2 + coefs[14]*x1*x2*x3 + coefs[15]*x1*x3*x3 + coefs[16]*x2*x2*x2 + coefs[17]*x2*x2*x3 + coefs[18]*x2*x3*x3 + coefs[19]*x3*x3*x3 

    elif n == 4:

        if m == 1:
        
            def fun(X, coefs):
                x1 = X[0]
                x2 = X[1]
                x3 = X[2]
                x4 = X[3]
                return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3  + coefs[4]*x4
            
        elif m == 2:
            
            def fun(X, coefs):
                x1 = X[0]
                x2 = X[1]
                x3 = X[2]
                x4 = X[3]
                return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3 + coefs[4]*x4 + coefs[5]*x1*x1 + coefs[6]*x1*x2 + coefs[7]*x1*x3 + coefs[8]*x1*x4 + coefs[9]*x2*x2 + coefs[10]*x2*x3 + coefs[11]*x2*x4 + coefs[12]*x3*x3 + coefs[13]*x3*x4 + coefs[14]*x4*x4
            
        elif m == 3:
            
            def fun(X, coefs):
            
                x1 = X[0]
                x2 = X[1]
                x3 = X[2]
                x4 = X[3]

                return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3 + coefs[4]*x4 + coefs[5]*x1*x1 + coefs[6]*x1*x2 + coefs[7]*x1*x3 + coefs[8]*x1*x4 + coefs[9]*x2*x2 + coefs[10]*x2*x3 + coefs[11]*x2*x4 + coefs[12]*x3*x3 + coefs[13]*x3*x4 + coefs[14]*x4*x4 + coefs[15]*x1*x1*x1 + coefs[16]*x1*x1*x2 + coefs[17]*x1*x1*x3 + coefs[18]*x1*x1*x4 + coefs[19]*x1*x2*x2 + coefs[20]*x1*x2*x3 + coefs[21]*x1*x2*x4 + coefs[22]*x1*x3*x3 + coefs[23]*x1*x3*x4 + coefs[24]*x1*x4*x4 + coefs[25]*x2*x2*x2 + coefs[26]*x2*x2*x3 + coefs[27]*x2*x2*x4 + coefs[28]*x2*x3*x3 + coefs[29]*x2*x3*x4 + coefs[30]*x2*x4*x4 + coefs[31]*x3*x3*x3 + coefs[32]*x3*x3*x4 + coefs[33]*x3*x4*x4 + coefs[34]*x4*x4*x4
            
        elif m == 4:
            
            def fun(X, coefs):
            
                x1 = X[0]
                x2 = X[1]
                x3 = X[2]
                x4 = X[3]

                return (coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3 + coefs[4]*x4 +  coefs[5]*x1*x1 +coefs[6]*x1*x2 + coefs[7]*x1*x3 + coefs[8]*x1*x4 + coefs[9]*x2*x2 + coefs[10]*x2*x3 + coefs[11]*x2*x4 +             coefs[12]*x3*x3 + coefs[13]*x3*x4 + coefs[14]*x4*x4 +           coefs[15]*x1*x1*x1 +coefs[16]*x1*x1*x2 + coefs[17]*x1*x1*x3 + coefs[18]*x1*x1*x4 +            coefs[19]*x1*x2*x2 + coefs[20]*x1*x2*x3 + coefs[21]*x1*x2*x4 +             coefs[22]*x1*x3*x3 + coefs[23]*x1*x3*x4 +             coefs[24]*x1*x4*x4 +             coefs[25]*x2*x2*x2 + coefs[26]*x2*x2*x3 +coefs[27]*x2*x2*x4 +             coefs[28]*x2*x3*x3 + coefs[29]*x2*x3*x4 +            coefs[30]*x2*x4*x4 +             coefs[31]*x3*x3*x3 + coefs[32]*x3*x3*x4 +             coefs[33]*x3*x4*x4 +            coefs[34]*x4*x4*x4 +            coefs[35]*x1*x1*x1*x1 + coefs[36]*x1*x1*x1*x2 +coefs[37]*x1*x1*x1*x3 + coefs[38]*x1*x1*x1*x4 +             coefs[39]*x1*x1*x2*x2 + coefs[40]*x1*x1*x2*x3 + coefs[41]*x1*x1*x2*x4 +             coefs[42]*x1*x1*x3*x3 + coefs[43]*x1*x1*x3*x4 +             coefs[44]*x1*x1*x4*x4 +             coefs[45]*x1*x2*x2*x2 + coefs[46]*x1*x2*x2*x3 + coefs[47]*x1*x2*x2*x4 +             coefs[48]*x1*x2*x3*x3 + coefs[49]*x1*x2*x3*x4 +             coefs[50]*x1*x2*x4*x4 +             coefs[51]*x1*x3*x3*x3 + coefs[52]*x1*x3*x3*x4 +             coefs[53]*x1*x3*x4*x4 +             coefs[54]*x1*x4*x4*x4 +             coefs[55]*x2*x2*x2*x2 + coefs[56]*x2*x2*x2*x3 + coefs[57]*x2*x2*x2*x4 +             coefs[58]*x2*x2*x3*x3 + coefs[59]*x2*x2*x3*x4 +             coefs[60]*x2*x2*x4*x4 +           coefs[61]*x2*x3*x3*x3 + coefs[62]*x2*x3*x3*x4 +             coefs[63]*x2*x3*x4*x4 +             coefs[64]*x2*x4*x4*x4 +             coefs[65]*x3*x3*x3*x3 + coefs[66]*x3*x3*x3*x4 +             coefs[67]*x3*x3*x4*x4 +             coefs[68]*x3*x4*x4*x4 +             coefs[69]*x4*x4*x4*x4        )
            
    return fun(X, coefs)      
##36, 39, 45, 49(HUGE?), 56, 58,61
#%%
# def main_fun_calculator(n, m, X, coefs):

#     return [allfun(n, m, [X[j][i] for j in range(n)], coefs) for i in range(len(X[0]))]


#%%

# def allfun(n, m, X, coefs):
#     def fun(*args):
#         result = coefs[0]
#         index = 1
        
#         for i in range(n):
#             result += coefs[index] * args[i]
#             index += 1
        
#         if m >= 2:
#             for i in range(n):
#                 for j in range(i, n):
#                     result += coefs[index] * args[i] * args[j]
#                     index += 1
        
#         if m == 3:
#             for i in range(n):
#                 for j in range(i, n):
#                     for k in range(j, n):
#                         result += coefs[index] * args[i] * args[j] * args[k]
#                         index += 1
        
#         return result

#     if isinstance(X[0], (list, tuple)):
#         Y = [fun(*X[i]) for i in range(len(X))]
#     else:
#         Y = fun(*X)
    
#     return Y
    
#%% important functions
def get_r2adj(y, x, method, err=None):
    """
    Fits and report a linear model ({method} = 'OLS' or 'WLS') for y(x) returning (r2_adj)
    """
    if method == 'OLS':
    
        model = sm.OLS(y, x).fit()
        
    elif method == 'WLS':
        
        weights =  1/(err**2)
        model = sm.WLS(y, x, weights=weights).fit()

    r2_adj = model.rsquared_adj

    return r2_adj 

def get_combinations_with_zero(lst):
    result = []
    for r in range(1, len(lst) + 1):
        for combo in combinations(lst, r):
            if 0 in combo:
                result.append(list(combo))
    return result


def find_max_r2adj(n, m, method, Y, X_poly_final_scaled, err=None):
    try_all = get_combinations_with_zero(get_allcoefs(n, m))
    
    r2_adjs = []
    selects = []

    # Use tqdm to create a progress bar
    for i in tqdm(try_all):
        r2_adj = get_r2adj(Y, X_poly_final_scaled[:, i], method, err)
        r2_adjs.append(r2_adj)
        selects.append(i)

    higher = [r2 for r2, select in zip(r2_adjs, selects) if r2 > max(r2_adjs) * 0.995]
    higher_selects = [select for r2, select in zip(r2_adjs, selects) if r2 > max(r2_adjs) * 0.995]
    
    print('====== Top 2% Adj-R2 ======')
    print(higher_selects)
    print(higher)
    print("=" * 50)
    
    ax = plt.gca()

    plt.scatter(np.arange(0, len(higher)), higher, color="#343E3D")
    plt.xticks(np.arange(0, len(higher_selects)), rotation=45)
    ax.set_xticklabels(higher_selects)
    plt.ylabel("Adjusted-$R^2$")
    
    return r2_adjs, selects



# find_max_r2adj(2, 3, 'OLS', 0, y1, X_poly_final_scaled)

def pareto_chart(n, m, coefs, r2_adj, select=None):
    
    fig = plt.figure()
    
    if select is None:
        select = list(get_allcoefs(n, m))
    else:
        select = select

    x_coefs = [f'x{i}' for i in select]
    effects = np.abs(coefs[coefs != 0])/np.sum(np.abs(coefs))*r2_adj
    
    combined = np.array([x_coefs, effects]).T
    print(combined)
    combined_sorted = combined[combined[:, 1].argsort()][::-1].T
    effects = combined_sorted[1].astype(float)
    cumsum = np.cumsum(effects)
    
    fig, ax1 = plt.subplots()
    
    ax1.bar(combined_sorted[0], effects, color=black)
    ax1.set_ylabel('Effect / %')
    # plt.xlabel()
    ax1.set_ylim(0,1)
    ax1.plot(combined_sorted[0], cumsum, color=red, ls='-', marker='o')

    return

def pred_actual(n, m, X_scaled, Y, coefs, model_final):
    
    fig = plt.figure()

    y_pred = allfun(n, m, X_scaled, coefs)
    
    x_y = np.linspace(Y.min(), Y.max(), 100)

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=Y, y=y_pred, mode='markers', line=dict(color='#343E3D', width=0)))
    # fig.add_trace(go.Scatter(x=x_y, y=x_y, mode='lines', line=dict(color='#FF5252', width=2, dash='dash')))

    # fig.update_layout(
    #     showlegend=False,
    #     font=dict(family='Lato', color='black', size=15),
    #     title='Predicted vs Actual',
    #     title_x=0.5,
    #     yaxis_title="Predicted",
    #     xaxis_title="Actual",     
    # )
    
    # fig.show()
    
    # fig = plt.figure()

    print(len(Y), len(y_pred))
    plt.scatter(Y, y_pred, color=black, s=15)
    plt.plot(x_y, x_y, linestyle='--', color=red)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    
    
    # fig = plt.figure()
    
    # plt.scatter(y_pred, model_final.resid/Y)
    
    return

def residuals_graph(resids, y=None):
    
    fig = plt.figure()

    plt.scatter(np.arange(0, len(resids)), resids, color=black)
    plt.axhline(y=0, ls='--', color=red)
    plt.ylabel('Residuals')
    plt.xlabel('')
    plt.xticks([])

        
    return

def main_regmodel(n, m, method, X, Y, err=None, select=None, test_all=False):
    
    X = pd.DataFrame(X).T

    X_final_scaled, X_poly_final_scaled, scaler = poly_scale(X, m)
    
    if select is None:
        select = list(get_allcoefs(n, m))
    else:
        select = select
        
    if method == 'OLS':
    
        if test_all is True:
            find_max_r2adj(n, m, 'OLS', Y, X_poly_final_scaled)
            
        model_final, coefs_sm, r2, r2_adj = reg_model( Y, X_poly_final_scaled[:,select], 'OLS')
        
    elif method == 'WLS':
        
        if test_all is True:
            find_max_r2adj(n, m, 'WLS', Y, X_poly_final_scaled, err)
            
        model_final, coefs_sm, r2, r2_adj = reg_model( Y, X_poly_final_scaled[:,select], 'WLS', err)  
        
    diff = get_allcoefs(n, m).difference(select)

    coefs = np.zeros(len(get_allcoefs(n, m)))
    
    for i in diff:
        coefs[i] = 0
    for i in np.arange(0, len(select)):
        coefs[select[i]] = coefs_sm[i]
        
    effects = np.abs(coefs)/(np.sum(np.abs(coefs))-coefs[0])*100
    
    pareto_chart(n, m, coefs, r2_adj, select)
    pred_actual(n, m, X_final_scaled.T, Y, coefs, model_final)
    residuals_graph(model_final.resid)
    
    print('')
    print("="*78)
    print('effects = ', effects)
    print("="*78)
    print('coefs = ', coefs)
    print("="*78)
    print('')
    
    return model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled, effects




#%%



def up_layout_surface(fig, x, y, z, title=None): 
    
    fig.update_layout(font_family='Lato', font_color='black', font_size=13,
        
        title=title,title_x = 0.5, 
        scene = dict(
            xaxis = dict(
                title=x,
                # tickvals=[7,7.5,8]
                ),
            yaxis = dict(
                title=y,
                # tickvals=[0.01, 0.05, 0.09]
                ),
            zaxis = dict(
                title=z,
                ),
          #   annotations=[
          # dict(
          #     showarrow=False,
          #     x=50,
          #     y=1,
          #     z=150,
          #     text="Point 1",
              # xanchor="left",
              # xshift=10,z
              # opacity=0.7),
            
          # )
        # ]
            ),
                     
        #plot_bgcolor='white'
        autosize=True,
        # margin=dict(l=10, r=20, t=10, b=10),
        #paper_bgcolor='#242424',
        # width= 550,height=550,
        legend=dict(
            x=0.005,
            y=.95,
            #traceorder="normal",
            font=dict(
                family="Lato",
                size=10,
                color="black"
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        #yaxis_range=[-.5, 0.5],
        #xaxis_range=[-.3, 1.1]
        
        
        )
    
def make_surface3D(x1, x2, x3, coefs, fun):
        
    surf = np.array(fun(np.ravel(x1), np.ravel(x2), np.ravel(x3), coefs))
    surface = surf.reshape((100,100))
        
    return surface

def make_surface3D(n, m, coefs, fixposition = None, fixvalue = None):
    
    scaled_X1, scaled_X2 = make_2D(0, 1, 0, 1, 100)
    
    X = [scaled_X1, scaled_X2]
    
    if type(fixposition) is int :
     
        X = [fixvalue,fixvalue,fixvalue]
    
        X[fixposition] = fixvalue
        
        if fixposition == 0:
            X[1], X[2] = scaled_X1, scaled_X2
        
        if fixposition == 1:
            X[0], X[2] = scaled_X1, scaled_X2
            
        if fixposition == 2:
            X[0], X[1] = scaled_X1, scaled_X2
            
    
    elif type(fixposition) is list:
        
        X = [fixvalue,fixvalue,fixvalue,fixvalue]
    
        if fixposition == [0, 1]:
            X[2], X[3] = scaled_X1, scaled_X2
        
        if fixposition == [0, 2]:
            X[1], X[3] = scaled_X1, scaled_X2
            
        if fixposition == [0, 3]:
            X[1], X[2] = scaled_X1, scaled_X2
            
        if fixposition == [1, 2]:
            X[0], X[3] = scaled_X1, scaled_X2
            
        if fixposition == [1, 3]:
            X[0], X[2] = scaled_X1, scaled_X2
        
        if fixposition == [2, 3]:
            X[0], X[1] = scaled_X1, scaled_X2

    return allfun(n, m, X, coefs)

def make_2D(min_x, max_x, min_y, max_y, step=100):
    """
    Create a 2D grid of points.

    Args:
        min_x (float): Minimum value of the x-axis.
        max_x (float): Maximum value of the x-axis.
        min_y (float): Minimum value of the y-axis.
        max_y (float): Maximum value of the y-axis.
        step (int): Number of steps for both x and y axes.

    Returns:
        [np.ndarray, np.ndarray]: A list of Two NumPy arrays representing the X and Y coordinates of the grid points.
    """
    x = np.linspace(min_x, max_x, step)
    y = np.linspace(min_y, max_y, step)
    X, Y = np.meshgrid(x, y)
    
    return [X, Y]

#%%
def main_surface(n, m, coefs, X, Y, fixposition=None, fixvalue = None, title=None, step=100, names=None):

    if fixposition is None:
        
        Z = make_surface3D(n, m, coefs)
        
        X_units = make_2D(X[0].min(), X[0].max(), X[1].min(), X[1].max())
        
        surface = plot_surface(X_units, Z, X, Y)
        
        up_layout_surface(surface, X[0].name, X[1].name, 'I', title)
        
        surface.show(rendered = 'browser')
        
    
    if type(fixposition) == int:

        ij = [[1,2], [0,2], [0,1]]

        for item in [0, 1, 2]:
            
            i = ij[item][0]
            j = ij[item][1]
            fixposition = item
            
            Z = make_surface3D(n, m, coefs, fixposition=fixposition, fixvalue=fixvalue)
            X_units = make_2D(X[i].min(), X[i].max(), X[j].min(), X[j].max())
            # X_units = make_2D(.25, .31, 2.9, 3.1)
            
            surface = plot_surface(X_units, Z, [X[i], X[j]], Y)
            up_layout_surface(surface, X[i].name, X[j].name, 'I', title)
            surface.show(rendered='browser')
            
            ## CONTOUR PLOT
            
            fig = plt.figure()
            levels = np.round(np.linspace(0.047, 0.12, 25), 3)
            cp = plt.contourf(X_units[0], X_units[1], Z, levels=levels, cmap='magma')
            # plt.title(f'[{i}, {j}]')
            fig.colorbar(cp, label='dE / V')
            
            if names == None:
                plt.xlabel(X[i].name)
            else:
                plt.xlabel(names[i])
            if names == None:     
                plt.ylabel(X[j].name)
            else:
                plt.ylabel(names[j])

    if type(fixposition) == list:

        ij = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        ji = [[2,3], [1,3], [1,2], [0,3], [0,2], [0,1]]

        for item in [0, 1, 2, 3, 4, 5]:
            
            i = ji[item][0]
            j = ji[item][1]
            fixposition = ij[item]
            
            Z = make_surface3D(n, m, coefs, fixposition=fixposition, fixvalue=fixvalue)
            X_units = make_2D(X[i].min(), X[i].max(), X[j].min(), X[j].max())
            surface = plot_surface(X_units, Z, [X[i], X[j]], Y)
            print(Z)
            print(np.round(np.linspace(Z.min()*0.95, Z.max()*1.05, 25), 3))
            
            up_layout_surface(surface, X[i].name, X[j].name, 'dE / V', title)
            # surface.show(rendered='browser')
            
            
            
            fig = plt.figure()
            levels = np.round(np.linspace(0.047, 0.12, 25), 3)
            # levels = np.round(np.linspace(Z.min()*0.95, Z.max()*1.05, 25), 3)
            # levels = np.round(np.linspace(-.05, 0, 25), 3)
            cp = plt.contourf(X_units[0], X_units[1], Z, levels=levels, cmap='magma')
            # plt.title(f'[{i}, {j}]')
            # fig.colorbar(cp, label='i$_p$ / $\mu$A')
            fig.colorbar(cp, label='dE / V')
            
            if names == None:
                plt.xlabel(X[i].name)
            else:
                plt.xlabel(names[i])
            if names == None:     
                plt.ylabel(X[j].name)
            else:
                plt.ylabel(names[j])
                
            for c in cp.collections:
                c.set_edgecolor("face")
                
            plt.savefig(f'fix\ip-{X[i].name}-{X[j].name}.pdf', bbox_inches='tight')
           
            # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            
            # ax = plt.gca()
            
            # surf = ax.plot_surface(X_units[0], X_units[1], Z, cmap=cm.magma,
            #             linewidth=0, antialiased=True)

            # # Customize the z axis.
            # ax.set_zlim(Z.min()*0.85, Z.max()*1.15)
            # # ax.zaxis.set_major_locator(LinearLocator(10))
            # # # A StrMethodFormatter is used automatically
            # # ax.zaxis.set_major_formatter('{x:.02f}')
            
            # # Add a color bar which maps values to colors.
            # ax.view_init(-165, 200)
            
            # fig.colorbar(surf, shrink=1, aspect=10)
            
          

    return 

#%%
def plot_surface(X_units, Z, X, Y):
    
    x1 = X[0]
    x2 = X[1]
    
    X1 = X_units[0]
    X2 = X_units[1]
    
    fig = go.Figure(data=[go.Surface(
        x=X1,
        y=X2,
        z=Z,
        name='d8',
        colorscale='bluyl',
        showscale=False,
    )])


    fig.add_trace(go.Scatter3d(
        x=x1,
        y=x2,
        z=Y,
        # name='d4',
        mode='markers',
        marker=dict(
            # size=12,
            color='#343E3D',                # set color to an array/list of desired values
            # colorscale='Viridis',   # choose a colorscale
            opacity=0.4
        )
    )
         
        )
    
    fig.update_layout(
        xaxis_title = 'freq',
        yaxis_title = 'step'
        )
    
    return fig

#%%

def train_test(n, m, coefs, X, Y, X_test, Y_test, fixposition=None, fixvalue = None, title=None, step=100):
    
    

    if fixposition is None:
        
        Z = make_surface3D(n, m, coefs)
        
        X_units = make_2D(X[0].min(), X[0].max(), X[1].min(), X[1].max())
        
        surface = plot_surface(X_units, Z, X, Y)
        
        up_layout_surface(surface, X[0].name, X[1].name, 'I', title)
        
        surface.show(rendered = 'browser')
        
    
    if type(fixposition) == int:

        ij = [[1,2], [0,2], [0,1]]

        for item in [0, 1, 2]:
            
            i = ij[item][0]
            j = ij[item][1]
            fixposition = item
            
            Z = make_surface3D(n, m, coefs, fixposition=fixposition, fixvalue=fixvalue)
            X_units = make_2D(X[i].min(), X[i].max(), X[j].min(), X[j].max())
            X_test = make_2D(X_test[i].min(), X_test[i].max(), X_test[j].min(), X_test[j].max())
            surface = plot_surface(X_units, Z, [X[i], X[j]], Y)
            
            surface.add_trace(go.Scatter3d(
                x=X_test[0],
                y=X_test[1],
                z=Y_test,
                # name='d4',
                mode='markers',
                marker=dict(
                    # size=12,
                    color='#343E3D',                # set color to an array/list of desired values
                    # colorscale='Viridis',   # choose a colorscale
                    opacity=0.4
                )
            )
                 
                )
            
            up_layout_surface(surface, X[i].name, X[j].name, 'I', title)
            surface.show(rendered='browser')

    if type(fixposition) == list:

        ij = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

        for item in [0, 1, 2, 3, 4, 5]:
            
            i = ij[item][0]
            j = ij[item][1]
            fixposition = ij[item]
            
            Z = make_surface3D(n, m, coefs, fixposition=fixposition, fixvalue=fixvalue)
            print(i, j )
            X_units = make_2D(X[i].min(), X[i].max(), X[j].min(), X[j].max())
            X_test2 = make_2D(X_test[i].min(), X_test[i].max(), X_test[j].min(), X_test[j].max())
            surface = plot_surface(X_units, Z, [X[i], X[j]], Y)
            up_layout_surface(surface, X[i].name, X[j].name, 'I', title)
            
            surface.add_trace(go.Scatter3d(
                x=X_test[i],
                y=X_test[j],
                z=Y_test,
                # name='d4',
                mode='markers',
                marker=dict(
                    # size=12,
                    color=red,                # set color to an array/list of desired values
                    # colorscale='Viridis',   # choose a colorscale
                    opacity=0.4
                )
            ))
            
            
            surface.show(rendered='browser')
          

    return

