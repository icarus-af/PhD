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

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

xlabel = 'Power'
xrange = [2.5, 3]
ylabel = 'Separation'
yrange = [15, 18]
zlabel = 'Height'
zrange = [7, 7]

df = build.full_fact(
    {xlabel: xrange,
     ylabel: yrange,
     zlabel: zrange,
     
     }
    
    
    )

#pc
df.loc[len(df)] = [(xrange[0]+xrange[1])/2, (yrange[0]+yrange[1])/2, (zrange[0]+zrange[1])/2]

#fix decimal
df[ylabel] = df[ylabel]/100

x = df[xlabel]
y = df[ylabel]
z = df[zlabel]


df = df.drop_duplicates()


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

l = list(range(len(df)))
random.shuffle(l)


df['order'] = l

print(df.sort_values(by=['order']))

df2=df