# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 18:44:24 2022

@author: 12esq
"""

from doepy import build
from doepy import read_write

import matplotlib.pyplot as plt
from matplotlib import rcParams

import random

import numpy as np

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

def add_cp(df, reps):
    
    means = [df.iloc[:, i].mean() for i in np.arange(len(df.T))]

    while reps != 0:
        reps=reps-1
        print(reps)
        df.loc[len(df)] = means
        
def random_order(df):
    
    df_random = df
    
    l = list(range(len(df)))
    random.shuffle(l)
    df_random['order'] = l

    print(df_random.sort_values(by=['order']))
    
    return df_random


#%%

xlabel = 'Power'
xrange = [2.5, 3]
ylabel = 'Separation'
yrange = [0.015, 0.018]
zlabel = 'Height'
zrange = [7, 11]
ulabel = 'Velocity'
urange = [20,100]

df = build.box_behnken(
    
    {xlabel: xrange,
     ylabel: yrange,
     zlabel: zrange,
     ulabel: urange,
     }
)

add_cp(df, 3)
df_random = random_order(df)


#%%


#%%


x = df[xlabel]
y = df[ylabel]
z = df[zlabel]

# read_write.write_csv(df, filename='pyDOE.csv')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, color='#343E3D')
plt.rcParams['grid.color'] = (0, 0, 0, 0)

ax.set_xlabel(xlabel)
# ax.set_xlim([x.min(), x.max()])
# ax.set_ylim([y.min(), y.max()])
# ax.set_zlim([z.min(), z.max()])
ax.set_xticks([x.min(), (xrange[0]+xrange[1])/2, x.max()])
ax.set_ylabel(ylabel)
ax.set_yticks([y.min(), (yrange[0]+yrange[1])/200, y.max()])
ax.set_zlabel(zlabel)
ax.set_zticks([z.min(), (zrange[0]+zrange[1])/2, z.max()])

# fig = go.Figure()

# fig.add_trace(go.Scatter3d(
#     x=df['Power'],
#     y=df['Separation'],
#     z=df['Height'],
#     # name='d4',
#     mode='markers',
#     marker=dict(
#         # size=12,
#         color='#343E3D',                # set color to an array/list of desired values
#         # colorscale='Viridis',   # choose a colorscale
#         opacity=0.4
#     ),
#     autosize=True
# )
     
#     )

# fig.update_layout(
#     xaxis_title = 'freq',
#     yaxis_title = 'step'
#     )

# fig.show()

