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

import data_treatment_model as dt

from tqdm import tqdm

#%%



df = pd.read_csv('20231005_treated.csv')


data = df[:-1]


x1 = data['freq']
x2 = data['amp']
x3 = data['step']


X = [x1, x2, x3]
Y = data['Ip-adj']

# errors = data['dE_std']
# err = np.array([df['dE}_std'].min()]*17)


err = 0

n = 3
m = 1

select=[0,1,2,3] ## m=1, R2=0.853, Ip
select=[0,1,2,8] ## m=2, R2=0.859, Ip

select=[0,1,2,3] ## m=1, R2=0.911, Ip-b
select=[0,1,2,8] ## m=2, R2=0.917, Ip-b, residualsOK

select=[0,1,2,3] ## m=1, R2=0.853, Ip-avg
select=[0,1,2,8] ## m=2, R2=0.917, Ip-avg, residualsOK


model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled, effects = dt.main_regmodel(n, m, 'OLS', X, Y, test_all=True, select=None)
#%%
dt.main_surface(n, m, coefs, X, Y, fixposition=0, fixvalue=0.5)

#%% TESTES COM IMAGENS

data = df


x1 = data['alt']
x2 = data['sep']
x3 = data['vel']
x4 = data['pot']
#%/data['mean_rep'].max()
#%%
Ys = []

tipos = ['mean', 'skew', 'sd']

for tipo in tipos:
    
    x = []
    
    Y = data[f'{tipo}_rep']
    err = data[f'{tipo}_std_rep']
    for i in range(len(Y)):
        yy = ufloat(Y[i], err[i])
        x.append(yy)
    
    Ys.append(x)
    
#%%

X = [x1,x3,x4]

combinado = pd.DataFrame(Ys).T
combinado.columns = tipos

combinado_norm = (combinado-combinado.min())/(combinado.max()-combinado.min())

combinado_norm['soma'] = combinado_norm.sum(axis=1)

Y = np.array([i.n for i in combinado_norm['soma']])
errors = np.array([i.s for i in combinado_norm['soma']])

# Y = np.array([i.n for i in Ys[0]])
# errors = np.array([i.s for i in Ys[0]])

n = 4
m = 2
# select = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
# 
select = [0,13,20,23,25,27,29,30] ## MODEL FOR dE!!!


# select = [0,2,27,30,34]
select = [0,1,2,3,4,5,6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
# select = [0,1,2,3,4,5,7,8,9,10,12,13,14,16,17,18,19,21,22,23,24,25,26,28,29,31,32,33]

# select = [0,4,8,9,11,22,24,26,30]
select = [0,13,20,23,25,27,29] ##FINAL FOR dE

# select = [0, 25, 27, 30]

select = [0,1,3,9,10,11] #means_hist, stdevs_hist (WLS) 0.729 SIG low
select = [0,9,11] #median_rep, median_std_rep (WLS) 0.462 SIG all
select = [0,2,4,8,11] #skew_rep, skew_std_rep (WLS) 0.692 SIG all
select = [0,4,10,11] #sd_rep, sd_std_rep (WLS) 0.857 SIG all
select = [0,9,11] # All combined (OLS) 0.588 SIG all
select = [0,9,11] # All combined with uncertainties (WLS) 0.834 SIG all

select = [0,4,10,11]

model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled = dt.main_regmodel(n, m, 'WLS', X, Y, err=errors, select=select, test_all=False)

#%%


Z = dt.main_surface(4,2,coefs,X,Y, fixposition=[0,1], fixvalue=1)
#%%
combined = Y
#%% dE(img)
data = df


x1 = data['mean_rep']
x2 = data['median_rep']
x3 = data['skew_rep']
x4 = data['sd_rep']

xs = x1+x3+x4

X = [x1, x3, x4]

# X=[combined]

Y = data['dE']
# combinado = Y
# Y = (combinado-combinado.min())/(combinado.max()-combinado.min())

errors = data['dE_std']

n = 3
m = 2

# select = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

select = [0,13,20,23,25,27,29] ## MODEL FOR dE(alt, sep, vel, pot)
select= [0,1,3,5,7,8,9] ## MODEL FOR dE(mean, skew, sd)

# select = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]


model_final, coefs, r2, r2_adj, X_final_scaled, _scaled, effects = dt.main_regmodel(n, m, 'WLS', X, Y, err=errors, select=select, test_all=True)

#%%

xxx = np.linspace(0, 1, 100)

yyy = coefs[0] + coefs[1]*(xxx**1) + coefs[2]*(xxx**2) + coefs[3]*(xxx**3)

plt.scatter(X,Y)

plt.plot(np.linspace(X[0].min(), X[0].max(), 100),yyy)
#%%

dps = [ [df['mean_rep'][i],df['median_rep'][i],df['skew_rep'][i],df['sd_rep'][i]] for i in range(len(df))] 

x = np.arange(0,len(df))

err = df['dE_std']
Y = df['dE']

fig=plt.figure()

plt.errorbar(x, Y, color='black', yerr=err, fmt='o', alpha=.5)
# plt.scatter(positions, Y_test_calculated, color='red')

plt.ylabel('dE')
plt.xlabel('[alt, sep, vel, pot]')

ax=plt.gca()

xtickslabels = [f'{str(i)}' for i in dps]
ax.set_xticks(x)
ax.set_xticks(ticks=x, labels=xtickslabels, rotation=90)

#%%


Z = dt.main_surface(n,m,coefs,X,Y,fixposition=0, fixvalue=0.5)



#%%

bro = pd.DataFrame(X_final_scaled)

bro = bro.sub(bro.mean(axis=0), axis=1)

bro = X
#%%
fig = go.Figure(data=( go.Scatter3d(
    x=bro[0],
    y=bro[1],
    z=bro[2],
    # name='d4',
    mode='markers',
    marker=dict(
        # size=12,
        color='#343E3D',                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        opacity=0.4
    )
)))

fig.show(rendered='browser')

#%%

train = df.sample(n=14)
test = df.drop(train.index)

x1_test = test['mean_rep']
x2_test = test['median_rep']
x3_test = test['skew_rep']
x4_test = test['sd_rep']


X_test = np.array([x1_test, x3_test, x4_test])
Y_test = test['dE']
errors_test = test['dE_std']

x1 = train['mean_rep']
x2 = train['median_rep']
x3 = train['skew_rep']
x4 = train['sd_rep']

X = [x1, x3, x4]
Y = train['dE']
errors = train['dE_std']

model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled = dt.main_regmodel(n, m, 'WLS', X, Y, err=errors, select=select)
    
X_test_final_scaled, X_test_poly_final_scaled, scaler = dt.poly_scale(X_test, m)

Y_test_calculated = dt.allfun(n, m, X_test_final_scaled, coefs)

resids = Y_test - Y_test_calculated

dt.residuals_graph(resids, Y_test)

dt.pred_actual(n, m, X_test_final_scaled, Y_test, coefs, model_final)

dps = [ [df['mean_rep'][i],df['skew_rep'][i],df['sd_rep'][i]] for i in range(len(df))] 

x_test = [list(i) for i in X_test.T]
checks = []
positions = []

for j in x_test:
    test = [i==j for i in dps]
    checks.append(test)

for i in range(len(checks)):
    
    pos = np.where(checks[i])[0][0]
    positions.append(pos)

x = np.arange(0,len(df))

err = df['dE_std']
Y = df['dE']

fig=plt.figure()

plt.errorbar(x, Y, color='black', yerr=err, fmt='o', alpha=.5)
plt.scatter(positions, Y_test_calculated, color='red')

plt.ylabel('dE')
plt.xlabel('[alt, sep, vel, pot]')

ax=plt.gca()

xtickslabels = [f'{str(i)}' for i in dps]
ax.set_xticks(x)
ax.set_xticks(ticks=x, labels=xtickslabels, rotation=90)

#%%



#%%

# def train_test(n, m, method, X, Y, err=None, select=None, test_all=False):
    
#     train = X.sample(n=round(len(df)*0.25),0)
#     test = X.drop(train.index)
#     #%%
# df = pd.read_csv('alldata-3.csv')

# x1 = df['alt']
# x2 = df['sep']
# x3 = df['vel']
# x4 = df['pot']

# data = pd.DataFrame([x1, x2, x3, x4]).T

# train = data.sample(n=int(len(X)-round(len(X)*pct,0)))
# test = df.drop(train.index)

#%%

df = pd.read_csv('alldata-3.csv')

n = 4
m = 3

train = df.sample(n=14)
test = df.drop(train.index)

x1_test = test['alt']
x2_test = test['sep']
x3_test = test['vel']
x4_test = test['pot']

X_test = np.array([x1_test, x2_test, x3_test, x4_test])
Y_test = test['means_hist']
errors_test = test['stdevs_hist']

x1 = train['alt']
x2 = train['sep']
x3 = train['vel']
x4 = train['pot']

X = [x1, x2, x3, x4]
Y = train['means_hist']

errors = train['stdevs_hist']

select = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

select=[0,8,9,11,13,14,18,20,21,23,24,25,27,29,30,32,33]
select = [0,9,11,13,20,23,25,27,29,30]
select = [0,13,20,23,25,27,29]
# select = [13,25,29]
# select = [13,25,29]
select = [0,1,3,9,10,11]

model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled = dt.main_regmodel(n, m, 'WLS', X, Y, err=errors, select=select)
    
X_test_final_scaled, X_test_poly_final_scaled, scaler = dt.poly_scale(X_test, 3)

Y_test_calculated = dt.allfun(n, m, X_test_final_scaled, coefs)

resids = Y_test - Y_test_calculated

dt.residuals_graph(resids, Y_test)

dt.pred_actual(n, m, X_test_final_scaled, Y_test, coefs, model_final)

dps = [ [df['alt'][i],df['sep'][i],df['vel'][i],df['pot'][i]] for i in range(len(df))] 


x_test = [list(i) for i in X_test.T]
checks = []
positions = []

for j in x_test:
    test = [i==j for i in dps]
    checks.append(test)
    
for i in range(len(checks)):
    
    pos = np.where(checks[i])[0][0]
    positions.append(pos)


x = np.arange(0,len(df))

err = df['stdevs_hist']
Y = df['means_hist']

fig=plt.figure()

plt.errorbar(x, Y, color='black', yerr=err, fmt='o', alpha=.5)
plt.scatter(positions, Y_test_calculated, color='red')

plt.ylabel('means_hist')
plt.xlabel('[alt, sep, vel, pot]')

ax=plt.gca()

xtickslabels = [f'{str(i)}' for i in dps]
ax.set_xticks(x)
ax.set_xticks(ticks=x, labels=xtickslabels, rotation=90)

# plt.savefig('traintest_meanhist_0.pdf', bbox_inches='tight')
#%%

Z = dt.main_surface(4,3,coefs,X,Y, fixposition=[1,3], fixvalue=0.5)


#%%

df = pd.read_csv('alldata-3.csv')


#%%
data = df

n = 4
dvar = 'dE'
dvar_err = 'dE_std'
columns = data.columns[0:n]
print('columns: ', columns)

X = np.array([data[column] for column in columns])
Y = data[dvar]

errors = 0
errors = data[dvar_err]

m = 1

allcoefs = np.array(list(dt.get_allcoefs(n,m)))

select = [0,13,20,23,25,27,29]     ## dE
select = allcoefs
# select = [0,2,3,4,13]                 ## i_red

# select = [0,20,25,27,49] #dE m=4 THIS ONE IS THE BEST
select=[0,2,3]

model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled, effects = dt.main_regmodel(n, m, 'WLS', X, Y, err=errors, select=select, test_all=False)

#%%
pct=0.75
n_vars=4
columns = df.columns[0:n_vars]
dvar = 'dE'
err_column = 'dE_std'

n=4
m=4
select = [0,20,25,27,49]
LS = 'WLS'
#%%

# Re-build datasets

n_train = round(len(df)*pct)

train = df.sample(n=n_train)
test = df.drop(train.index).reset_index()
train = train.reset_index()

X_test = np.array([test[column] for column in columns])
X_train = np.array([train[column] for column in columns])

Y_test = test[dvar]
errors_test = test[err_column]

Y_train = train[dvar]
errors_train = train[err_column]


# Re-train model
model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled, effects = dt.main_regmodel(n, m, LS, X_train, Y_train, err=errors_train, select=select, test_all=False)


# All data figure

df_all = df
Y_all = Y
errors_all = errors

lenght = len(df_all)

X_test_final_scaled, X_test_poly_final_scaled, scaler = dt.poly_scale(X_test, m)
Y_test_calculated = dt.allfun(n, m, X_test_final_scaled, coefs)
resids = Y_test - Y_test_calculated
SSE = (resids**2).sum()
SST = ((Y_test - Y_test.mean())**2).sum()
r2_traintest = 1 - SSE/SST

dps = [[df_all[column][i] for column in columns] for i in range(lenght)]

x = np.arange(0, lenght)
test_positions = test['index']

fig=plt.figure()

plt.errorbar(x, Y_all, color='black', yerr=errors_all, fmt='o', alpha=.8, capsize=2)
plt.scatter(test_positions, Y_test_calculated, color='red')

plt.ylabel(dvar)
plt.xlabel([i for i in columns])

ax=plt.gca()

xtickslabels = [f'{str(i)}' for i in dps]
ax.set_xticks(x)
ax.set_xticks(ticks=x, labels=xtickslabels, rotation=90)

# plt.savefig('alldata-mean_hist.pdf', bbox_inches='tight'

# Print results

print('SSE', 'r2_traintest')
print(SSE, r2_traintest)


#%%

Z = dt.main_surface(4,4,coefs,X,Y, fixposition=[1,3], fixvalue=[0.77, 0.77], names=['Altura / mm', 'Separação / mm', 'Velocidade / mm s$^{-1}$', 'Potência / W'])

#%%
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('alldata-3-ALL.csv')

data = df

n = 4
dvar = 'dE'
dvar_err = 'dE_std'
columns = data.columns[1:n+1]

X = np.array([data[column] for column in columns])
Y = data['dE']

X_final_scaled = dt.scale(X).T
#%%
knnreg = KNeighborsRegressor(n_neighbors = 3).fit(X_final_scaled, Y)

y_pred = knnreg.predict(X_final_scaled)

print(r2_score(Y, y_pred))
print()
#%%

X_train, X_test, y_train, y_test = train_test_split(X_final_scaled, Y)

knnreg = KNeighborsRegressor(n_neighbors = 3).fit(X_train, y_train)

print(knnreg.predict(X_test))
print(knnreg.score(X_test,y_test))

#%%
dps = [ [df['alt'][i],df['sep'][i],df['vel'][i],df['pot'][i]] for i in range(len(df))] 

x = np.arange(0,len(df))

item = 'dE'

err = df[f'{item}_std']
Y = df[f'{item}']

fig=plt.figure()

plt.errorbar(x, Y, color='black', yerr=err, fmt='o', alpha=.8, capsize=2)
# plt.scatter(x, Y, color='black')
# plt.scatter(positions, Y_test_calculated, color='red')

plt.ylabel('dE')
plt.xlabel('[alt, sep, vel, pot]')
plt.ylim([0.05, 0.11])

ax=plt.gca()

xtickslabels = [f'{str(i)}' for i in dps]
ax.set_xticks(x)
ax.set_xticks(ticks=x, labels=xtickslabels, rotation=90)

# plt.savefig('alldata-mean_hist.pdf', bbox_inches='tight')


#%%


coefs_str =  ['intercept' ,  'x1'  , 'x2'  ,  'x3'  ,  'x4'  ,  'x1*x1'  ,  'x1*x2'  ,  'x1*x3'  ,  'x1*x4' , 'x2*x2'  , 'x2*x3'  ,  'x2*x4'  ,  'x3*x3'  ,  'x3*x4'  , 'x4*x4'  , 'x1*x1*x1'  ,  'x1*x1*x2'  ,  'x1*x1*x3'  ,  'x1*x1*x4'  ,  'x1*x2*x2'  , 'x1*x2*x3'  ,  'x1*x2*x4'  , 'x1*x3*x3'  , 'x1*x3*x4'  , 'x1*x4*x4'  , 'x2*x2*x2'  ,  'x2*x2*x3'  ,  'x2*x2*x4'  , 'x2*x3*x3'  , 'x2*x3*x4'  ,  'x2*x4*x4'  , 'x3*x3*x3'  ,  'x3*x3*x4'  ,  'x3*x4*x4'  , 'x4*x4*x4']

# coefs_str =  ['intercept' ,  'x1'  , 'x2'  ,  'x3'  ,  'x1*x1'  ,  'x1*x2'  ,  'x1*x3'  , 'x2*x2'  , 'x2*x3'  ,  'x3*x3'  ]

coefs_str_ = pd.DataFrame(coefs_str)[coefs != 0]

df2 = coefs_str_.replace('x1','alt', regex=True)
df2 = df2.replace('x2','sep', regex=True)
df2 = df2.replace('x3','vel', regex=True)
df2 = df2.replace('x4','pot', regex=True)

#%%

eff = effects[effects != 0]
#%%
from matplotlib.lines import Line2D

names = np.array(df2[0])

c = [black, black, red, red, black, red]

plt.bar(np.arange(1,7), eff[1:], color=c)

ax = plt.gca()

plt.ylim(0,30)

ax.set_ylabel('Effects / %')
ax.set_xticklabels(['0','vel*pot', 'alt*vel*sep', 'alt*vel*pot', 'sep$^3$', 'sep$^2$pot', 'sep*vel*pot'], rotation=20)



legend_elements = [Line2D([0],[0], marker='o', ls='', color=black, label='Positive', markersize=10), Line2D([0],[0], marker='o', ls='', color=red, label='Negative', markersize=10)]

plt.legend(handles=legend_elements, frameon=False)

# cumsum = np.cumsum(eff[1:])

# plt.plot(df2[0][1:], cumsum, color=red, ls='-', marker='o')

plt.savefig('effects.pdf', bbox_inches='tight')



#%%
for i in range(len(coefs_str_)):
    print(coefs[coefs != 0][i], '*', np.array(df2).ravel()[i], ' + ') 
    
#%%

df = pd.read_csv('alldata-3.csv')

n = 4
m = 3

train = df.sample(n=14)
test = df.drop(train.index)

x1_test = test['alt']
x2_test = test['sep']
x3_test = test['vel']
x4_test = test['pot']

X_test = np.array([x1_test, x2_test, x3_test, x4_test])
Y_test = test['dE']
errors_test = test['dE_std']

x1 = train['alt']
x2 = train['sep']
x3 = train['vel']
x4 = train['pot']

X = [x1, x2, x3, x4]
Y = train['dE']

errors = train['dE_std']

select = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

select=[0,8,9,11,13,14,18,20,21,23,24,25,27,29,30,32,33]
select = [0,9,11,13,20,23,25,27,29,30]
select = [0,13,20,23,25,27,29]
# select = [13,25,29]
# select = [13,25,29]


model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled = dt.main_regmodel(n, m, 'WLS', X, Y, err=errors, select=select)
    
X_test_final_scaled, X_test_poly_final_scaled, scaler = dt.poly_scale(X_test, 3)

Y_test_calculated = dt.allfun(n, m, X_test_final_scaled, coefs)

resids = Y_test - Y_test_calculated

dt.residuals_graph(resids, Y_test)

dt.pred_actual(n, m, X_test_final_scaled, Y_test, coefs, model_final)

dps = [ [df['alt'][i],df['sep'][i],df['vel'][i],df['pot'][i]] for i in range(len(df))] 


x_test = [list(i) for i in X_test.T]
checks = []
positions = []

for j in x_test:
    test = [i==j for i in dps]
    checks.append(test)
    
for i in range(len(checks)):
    
    pos = np.where(checks[i])[0][0]
    positions.append(pos)


x = np.arange(0,len(df))

err = df['dE_std']
Y = df['dE']

fig=plt.figure()

plt.errorbar(x, Y, color='black', yerr=err, fmt='o', alpha=.5)
plt.scatter(positions, Y_test_calculated, color='red')

plt.ylabel('dE')
plt.xlabel('[alt, sep, vel, pot]')

ax=plt.gca()

xtickslabels = [f'{str(i)}' for i in dps]
ax.set_xticks(x)
ax.set_xticks(ticks=x, labels=xtickslabels, rotation=90)

# plt.savefig('traintest_meanhist_0.pdf', bbox_inches='tight')
# 



#%%
fig = dt.main_surface(4,3,coefs,X,Y, fixposition=[1,3], fixvalue=1)

#%%

dt.train_test(4, 3, coefs, X, Y, X_test, Y_test, fixposition=[1,3], fixvalue=1)

#%%

#%% TEST TRAIN BROOOOO IMG with dE
data = df

X = [x1+4,x2+4,x3+4,x4+4]
Y = data['dE']

errors = data['dE_std']
# err = np.array([df['dE}_std'].min()]*17)


err = 0

n = 4
m = 2

# select = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
# 
select = [0,13,20,23,25,27,29,30] ## MODEL FOR dE!!!


# select = [0,2,27,30,34]
select = [0,2,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
# select = [0,1,2,3,4,5,7,8,9,10,12,13,14,16,17,18,19,21,22,23,24,25,26,28,29,31,32,33]

# select = [0,4,8,9,11,22,24,26,30]
select = [0,13,20,23,25,27,29]

# select = [0, 25, 27, 30]

select= [0,2,3,5,6,7,8,9,10,11,12,14]
# select= [0,2,3,5,6,7,8,9,10,11,12]
# select= [0,6,5,9,8,11,10]
# select = [0, 1, 2, 4, 13]
model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled = dt.main_regmodel(n, m, 'WLS', X, Y, err=errors, select=select, test_all=False)
#%%
df = pd.read_csv('alldata-3.csv')

n = 4
m = 2

train = df.sample(n=14)
test = df.drop(train.index)

x1_test = test['mean_rep']
x2_test = test['median_rep']
x3_test = test['skew_rep']
x4_test = test['sd_rep']



X_test = np.array([x1_test, x2_test, x3_test, x4_test])
Y_test = test['dE']
errors_test = test['dE_std']

x1 = train['mean_rep']
x2 = train['median_rep']
x3 = train['skew_rep']
x4 = train['sd_rep']

X = [x1, x2, x3, x4]
Y = train['dE']

errors = train['dE_std']

# select = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

# select= [0,2,3,5,6,7,8,9,10,11,12,14]

model_final, coefs, r2, r2_adj, X_final_scaled, X_poly_final_scaled = dt.main_regmodel(n, m, 'WLS', X, Y, err=errors, select=None)
    
X_test_final_scaled, X_test_poly_final_scaled, scaler = dt.poly_scale(X_test, m)

Y_test_calculated = dt.allfun(n, m, X_test_final_scaled, coefs)

resids = Y_test - Y_test_calculated

# dt.residuals_graph(resids, Y_test)

# dt.pred_actual(n, m, X_test_final_scaled, Y_test, coefs, model_final)

#%%


dps = [ [df['alt'][i],df['sep'][i],df['vel'][i],df['pot'][i]] for i in range(len(df))] 


x_test = [list(i) for i in X_test.T]
checks = []
positions = []

for j in x_test:
    test = [i==j for i in dps]
    checks.append(test)
    #%%
for i in range(len(checks)):
    
    pos = np.where(checks[i])[0][0]
    positions.append(pos)

x = np.arange(0,len(df))

err = df['dE_std']
Y = df['dE']

fig=plt.figure()

plt.errorbar(x, Y, color='black', yerr=err, fmt='o', alpha=.5)
plt.scatter(positions, Y_test_calculated, color='red')

plt.ylabel('dE')
plt.xlabel('[alt, sep, vel, pot]')

ax=plt.gca()

xtickslabels = [f'{str(i)}' for i in dps]
ax.set_xticks(x)
ax.set_xticks(ticks=x, labels=xtickslabels, rotation=90)

# plt.savefig('traintest_meanhist_0.pdf', bbox_inches='tight')
# 

