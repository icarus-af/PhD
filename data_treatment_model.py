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

from uncertainties import ufloat
from uncertainties.umath import *

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Lato']
rcParams['axes.labelpad'] = 15
plt.rcParams['font.size'] = 12

## 6 cores ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF']

## 6 cores ['preto', 'vermelho', 'amarelo', 'verde', 'azul', 'rosa']

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


def poly_scale(df, degree):
    
    poly = PolynomialFeatures(degree=degree)
    scaler = MinMaxScaler()
    
    df_scaled = scaler.fit_transform(df)
    
    df_poly_scaled = poly.fit_transform(df_scaled)
    
    return df_scaled, df_poly_scaled, scaler

def reg_model(y, x):
    
    model = sm.OLS(y, x).fit()
    coefs = np.array(model.params)
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    
    return model, coefs, r2, r2_adj

def reg_model_WLS(y, x, err):
    
    weights =  1/(err**2)
    
    model = sm.WLS(y, x, weights=weights).fit()
    coefs = np.array(model.params)
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    
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

def data_table(table1, table2, table3):  
    
    

    return dash_table.DataTable(columns=[{"name": i, "id": i} for i in table1.columns],
                      data=table1.to_dict('records'),
                      
                      style_cell={'textAlign': 'center', 'backgroundColor': '#343E3D', 'color': '#FFFFFF', 'fontWeight': 'bold'},
                      style_header={'textAlign': 'center', 'backgroundColor': 'black'},
                      

                      
                         ),
    dash_table.DataTable(columns=[{"name": i, "id": i} for i in table2.columns],
                      data=table2.to_dict('records') ,
                      
                      style_cell={'textAlign': 'center', 'backgroundColor': '#343E3D', 'color': '#FFFFFF', 'fontWeight': 'bold'},
                      style_header={'textAlign': 'center', 'backgroundColor': 'black'},
                 
                      
                      style_data_conditional=(
                          
                          [
                              {
                                  'if': {
                                      'filter_query': '{P>|t|} > 0.05',
                                      
                                  },
                              'color':'#FF5252'
                              
                              },
                                          
                          
                          ]
                          
                          )  
                         ),
    dash_table.DataTable(columns=[{"name": i, "id": i} for i in table3.columns],
                      data=table3.to_dict('records'), 
                      style_cell={'textAlign': 'center', 'backgroundColor': '#343E3D', 'color': '#FFFFFF', 'fontWeight': 'bold'},
                      style_header={'textAlign': 'center', 'backgroundColor': 'black'},
                         
                      
                        )
    

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

allcoefs_3 = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}
allcoefs_2 = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14}
allcoefs_3_2 = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9}


def fun1_1(x1, coefs):
    z = coefs[0] + coefs[1]*x1
    return z

def fun1_2(x1, coefs):
    z = coefs[0] + coefs[1]*x1 + coefs[2]*x1*x1
    return z

def fun2_2(x,y, coefs):
    return coefs[0] + coefs[1]*x + coefs[2]*y + coefs[3]*x*x + coefs[4]*x*y + coefs[5]*y*y

def fun2_3(x,y, coefs):
    z = coefs[0] + coefs[1]*x + coefs[2]*y + coefs[3]*x*x + coefs[4]*x*y + coefs[5]*y*y + coefs[6]*x*x*x + coefs[7]*x*x*y + coefs[8]*x*y*y + coefs[9]*y*y*y
    return z

def fun3_1(x1, x2, x3, coefs):
    z = coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3
    return z
  
def fun_3_2(x1,x2,x3,coefs):
    return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3 + coefs[4]*x1*x1 + coefs[5]*x1*x2 + coefs[6]*x1*x3 + coefs[7]*x2*x2 + coefs[8]*x2*x3 + coefs[9]*x3*x3

def fun_3_3(x1,x2,x3,coefs):
    return  coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3 + coefs[4]*x1*x1 + coefs[5]*x1*x2 + coefs[6]*x1*x3 + coefs[7]*x2*x2 + coefs[8]*x2*x3 + coefs[9]*x3*x3 + coefs[10]*x1*x1*x1 + coefs[11]*x1*x1*x2 + coefs[12]*x1*x1*x3 + coefs[13]*x1*x2*x2 + coefs[14]*x1*x2*x3 + coefs[15]*x1*x3*x3 + coefs[16]*x2*x2*x2 + coefs[17]*x2*x2*x3 + coefs[18]*x2*x3*x3 + coefs[19]*x3*x3*x3 

def fun_4_2(x1, x2, x3, x4, coefs):
    return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3 + coefs[4]*x4 + coefs[5]*x1*x1 + coefs[6]*x1*x2 + coefs[7]*x1*x3 + coefs[8]*x1*x4 + coefs[9]*x2*x2 + coefs[10]*x2*x3 + coefs[11]*x2*x4 + coefs[12]*x3*x3 + coefs[13]*x3*x4 + coefs[14]*x4*x4
 
def fun_4_3(x1, x2, x3, x4, coefs):
        return coefs[0] + coefs[1]*x1 + coefs[2]*x2 + coefs[3]*x3 + coefs[4]*x4 + coefs[5]*x1*x1 + coefs[6]*x1*x2 + coefs[7]*x1*x3 + coefs[8]*x1*x4 + coefs[9]*x2*x2 + coefs[10]*x2*x3 + coefs[11]*x2*x4 + coefs[12]*x3*x3 + coefs[13]*x3*x4 + coefs[14]*x4*x4 + coefs[15]*x1*x1*x1 + coefs[16]*x1*x1*x2 + coefs[17]*x1*x1*x3 + coefs[18]*x1*x1*x4 + coefs[19]*x1*x2*x2 + coefs[20]*x1*x2*x3 + coefs[21]*x1*x2*x4 + coefs[22]*x1*x3*x3 + coefs[23]*x1*x3*x4 + coefs[24]*x1*x4*x4 + coefs[25]*x2*x2*x2 + coefs[26]*x2*x2*x3 + coefs[27]*x2*x2*x4 + coefs[28]*x2*x3*x3 + coefs[29]*x2*x3*x4 + coefs[30]*x2*x4*x4 + coefs[31]*x3*x3*x3 + coefs[32]*x3*x3*x4 + coefs[33]*x3*x4*x4 + coefs[34]*x4*x4*x4
    
#%%

df = pd.read_csv('DOE-opt-swv-3Dpen.csv')

# central = pd.DataFrame([df.loc[3], df.loc[4]])
# mean_central = central['I'].mean()
# std_central = central['I'].std()

# df = df.drop([0,3,4]).reset_index(drop=True)
# central_ = pd.DataFrame([[40, 110, 13, mean_central]], columns=['freq', 'amp', 'step', 'I'])
# df = pd.concat([df, central_], ignore_index=True)
# df['std'] = std_central

# from doepy import read_write
# # read_write.write_csv(df ,filename='DOE-opt-swv-3Dpen-final.csv')

# print(df)

def fun3(x,y, coefs):
    z = coefs[0] + coefs[1]*x*x + coefs[2]*x*x*x + coefs[3]*x*x*y
    return z

#%%
data = df


x1 = data['freq']
x2 = data['amp']
x3 = data['step']


y1 = data['I']
# err = data['std']
# err = np.array([df['dE_std'].min()]*17)
# y1 = y1/y1.max()

X_final = pd.DataFrame([x1, x2]).T

X_final_scaled, X_poly_final_scaled, scaler = poly_scale(X_final, 3)

# select = [0,1,2]
select = [0,3,6,7]
# select = list([0,1,4,5]) ## todos
# select = list([0,2,3,4,5,6,7,8,9,10,12,13,14,16,17,18,19])


model_final, coefs_sm, r2, r2_adj = reg_model( y1, X_poly_final_scaled[:,select])
# model_final, coefs_sm, r2, r2_adj = reg_model_WLS( y1, X_poly_final_scaled[:,select], err)

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

    
# preds=ANOVA(data, 'dE', 3, funop, X_final_scaled, coefs)
    
# summary_table1, summary_table2, summary_table3 = get_summaries(model_final)

#%%
data = df


x1 = data['freq']
x2 = data['amp']
# x3 = data['step']


y1 = data['I']
# err = data['std']
# err = np.array([df['dE_std'].min()]*17)
# y1 = y1/y1.max()

X_final = pd.DataFrame([x1, x2]).T

X_final_scaled, X_poly_final_scaled, scaler = poly_scale(X_final, 3)

select = [0,1,2]
select = [0,3,6,7]
# select = list([0,3,6,7]) ## todos


model_final, coefs_sm, r2, r2_adj = reg_model( y1, X_poly_final_scaled[:,select])
# model_final, coefs_sm, r2, r2_adj = reg_model_WLS( y1, X_poly_final_scaled[:,select], err)

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

    
# preds=ANOVA(data, 'dE', 3, funop, X_final_scaled, coefs)
    
# summary_table1, summary_table2, summary_table3 = get_summaries(model_final)

#%% pred x actual
x_scaled = X_final_scaled.T

y_pred = fun_3_2(x_scaled[0], x_scaled[1], x_scaled[2], coefs_sm)
x_y = np.linspace(y1.min(), y1.max(), 100)

pred_actual = go.Figure(data=[go.Scatter(x=y1, y=y_pred, line=dict(color='#343E3D', width=0), mode='markers'
    
    )
    
    ])

pred_actual.add_trace(go.Scatter(x=x_y, y=x_y, mode='lines', line=dict(color='#FF5252', width=2, dash='dash'), 
    
    ))

# model_residuals.add_hline(y=0, line_width=3, line_color='#FF5252', line_dash='dash', opacity=0.8)

pred_actual.update_layout(showlegend = False,font_family='Lato', font_color='black', font_size=15,
                          
                        
    title='Predicted vs Actual',title_x = 0.5,        
    yaxis_title="Predicted",
    xaxis_title="Actual",
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
    # yaxis_range=[-y_pred.max(), y_pred.max()],
    #yaxis_range=[-.5, 0.5],
    #xaxis_range=[-.3, 1.1]
    
    )

pred_actual.show()  

#%% r2s
var = np.arange(0+1, 9+1, 1)
r2s = []
r2s_adj = []

x_coefs = []

for item in np.arange(0,14+1,1):
    x_coefs.append('x{}'.format(item))
print(x_coefs)

for item in var:
    
    select = np.arange(0, item, 1)
    model, coefs, r2, r2_adj = reg_model_WLS( y1, X_poly_final_scaled[:,select], err)
    r2s.append(r2)
    r2s_adj.append(r2_adj)
    
    
    
r2s = pd.DataFrame(r2s, columns=['r2s'])
r2s['r2s_inc'] = r2s.r2s.diff().shift(-1)

r2s_adj = pd.DataFrame(r2s_adj, columns=['r2s_adj'])
r2s_adj['r2s_adj_inc'] = r2s_adj.r2s_adj.diff().shift(-1)

    
r2s_fig = go.Figure(data=[go.Bar(name='$R^2$', x=x_coefs[1:] ,y = r2s['r2s_inc'], marker_color='#343E3D' )
                        
                        ])

r2s_fig.add_trace(go.Bar(name='$R^2_{adj}$', x=x_coefs[1:] ,y = r2s_adj['r2s_adj_inc'], marker_color='#FF5252' ))
# correntes.add_trace(go.Bar(name='8 mM', x=x ,y = df['glic4']/df['glic8'], marker_color='#FFCE54' ))


r2s_fig.update_layout(font_family='Lato', font_color='black', font_size=9,
                          
                        
    title='',title_x = 0.5,        
    yaxis_title="Incremento",
    xaxis_title="Variável",
    plot_bgcolor='white',
    autosize=True,
    margin=go.Margin(l=0, r=0, t=0, b=0),
    #paper_bgcolor='#242424',
    
    legend=dict(
        x=0.005,
        y=.95,
        #traceorder="normal",
        font=dict(
            family="Lato",
            size=9,
            color="black"
        ),
        bgcolor='rgba(0,0,0,0)'
    ),
    #yaxis_range=[-.5, 0.5],
    #xaxis_range=[-.3, 1.1]
    
    )

r2s_fig.update_xaxes(showgrid = False, showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside' )
r2s_fig.update_yaxes(showgrid = False,showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside')

r2s_fig.show()

#%% effects

effects = np.abs(coefs_sm)/np.sum(np.abs(coefs_sm))
print(effects)
print(np.sum(coefs))
print(np.sum(effects))

x_coefs = []

for item in np.arange(0,35,1):
    x_coefs.append('x{}'.format(item))
print(x_coefs)

effects_fig = go.Figure(data=[go.Bar(name='4 mM', x=x_coefs ,y = effects*100, marker_color='#343E3D' )
                        
                        ])

# correntes.add_trace(go.Bar(name='8 mM', x=x ,y = df['glic8'], marker_color='#FF5252' ))
# correntes.add_trace(go.Bar(name='8 mM', x=x ,y = df['glic4']/df['glic8'], marker_color='#FFCE54' ))

effects_fig.add_hline(y=5, line=dict(color='#FF5252', width=2), opacity=.5 )

effects_fig.update_layout(font_family='Lato', font_color='black', font_size=8,
                          
                        
    title='',title_x = 0.5,        
    yaxis_title="Efeitos Padronizados / %",
    xaxis_title="Coeficientes",
    plot_bgcolor='white',
    autosize=True,
    margin=go.Margin(l=0, r=0, t=0, b=0),
    #paper_bgcolor='#242424',
    
    legend=dict(
        x=0.005,
        y=.95,
        #traceorder="normal",
        font=dict(
            family="Lato",
            size=8,
            color="black"
        ),
        bgcolor='rgba(0,0,0,0)'
    ),
    #yaxis_range=[-.5, 0.5],
    #xaxis_range=[-.3, 1.1]
    
    )

effects_fig.update_xaxes(showgrid = False, showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside' )
effects_fig.update_yaxes(showgrid = False,showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside')

effects_fig.show()

# effects_fig.write_image("images/effects_fig.pdf")

#%%

def plot_surface(X, Y, Z, x1, x2, y1):
    
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        name='d8',
        colorscale='bluyl',
        showscale=False,
    )])


    fig.add_trace(go.Scatter3d(
        x=x1,
        y=x2,
        z=y1,
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

def up_layout_surface(fig, title, x, y, z): 
    
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
              # xshift=10,
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

def make_2D(min_x, max_x, min_y, max_y, step):
    
    x = np.linspace(min_x, max_x, step)
    y = np.linspace(min_y, max_y, step)
    X, Y = np.meshgrid(x, y)
    
    return X, Y

#%%
def funop(x1,x2,x3,coefs):
    
    return coefs[0] + coefs[1]*x1*x1 + coefs[2]*x1*x1*x1 + coefs[3]*x1*x1*x2

# def fun3(x,y, coefs):
#     z = coefs[0] + coefs[3]*x1*x1 + coefs[6]*x1*x1*x1 + coefs[7]*x1*x1*x2
#     return z

# def funop(x1,x2,x3,coefs):
#     return coefs[0] + coefs[1]*x1*x2 + coefs[2]*x2*x2


X_g, Y_g = make_2D(0, 1, 0, 1, 100)

Z_FA = make_surface3D(X_g, Y_g,1, coefs_sm, funop)

XX_freq, XX_amp = make_2D(30, 70, 90, 170, 100)

x1 = df['freq']
x2 = df['amp']
# x3 = df['step']

surface = plot_surface(XX_freq, XX_amp, Z_FA, x1, x2, y1)
up_layout_surface(surface, '', 'freq', 'amp', 'I')
# surface = make_surface3D(X_final_scaled.T[0], X_final_scaled.T[1], X_final_scaled.T[2], coefs_sm, fun_3_2)
    
# fig = plot_surface(, df['amp'], y1, df['freq'], df['amp'], surface)

surface.show(rendered = 'browser')

    #%%

Z_FS = make_surface3D(X_g, .5, Y_g, coefs_sm, fun_3_2)

XX_freq, XX_step = make_2D(30, 50, 11, 15, 100)

surface = plot_surface(XX_freq, XX_step, Z_FS, x1, x3, y1)

surface.show(rendered = 'browser')

#%%

Z_AS = make_surface3D(.5, X_g, Y_g, coefs_sm, fun_3_2)

XX_amp, XX_step = make_2D(90, 130, 11, 15, 100)

surface = plot_surface(XX_amp, XX_step, Z_AS, x2, x3, y1)

surface.show(rendered = 'browser')