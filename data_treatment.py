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

#%%

def getcm(n, cmap):
    """
    Gets a sequence of n colors from cmap
    """
    
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    
    return colors

def plot(item, sep, tech, scan, b, Emax, Emin):
    """
    Reads an {item}.txt file filtered by {scan} column and outputs {peak_o, peak_r, ioveri, dEp}
    Different conditions based on {tech} = 'cv' or 'swv'
    if b           = 'b' then blank is considered on analysis
    Emax and Emin  = potential ranges for peak detection
    """
    
    # Read data for Cyclic Voltammetry analysis
    if tech == 'cv':
    
        df = pd.read_csv("{}.txt".format(item), sep=sep)
        
        if 'Scan' in df.columns:
        
            df = df[df['Scan'] == scan]
        
        E = df['WE(1).Potential (V)']
        i = df['WE(1).Current (A)']*1e6
        i_f = i
          
        if b == 'b':
            
            df_b = pd.read_csv("{}-b.txt".format(item), sep=sep)
            
            if 'Scan' in df_b.columns:
            
                df_b = df_b[df_b['Scan'] == scan]
                
            i_b = df_b['WE(1).Current (A)']*1e6
            i_f = i - i_b
            
    # Read data for Squared Wave Voltammetry analysis
    
    elif tech == 'swv':
        
        df = pd.read_csv("{}.txt".format(item), sep=sep)
        
        if 'Scan' in df.columns:
        
            df = df[df['Scan'] == scan]
        
        E = df['Potential applied (V)']
        i = df['WE(1).δ.Current (A)']*1e6
        i_f = i
          
        if b == 'b':
            
            df_b = pd.read_csv("{}-b.txt".format(item), sep=sep)
            
            if 'Scan' in df_b.columns:
            
                df_b = df_b[df_b['Scan'] == scan]
                
            i_b = df_b['WE(1).δ.Current (A)']*1e6
            i_f = i - i_b
            
    # Calculate pertinent data analysis
           
    peak_o = i_f[E < Emax][E > Emin].max()
    peak_r = i_f[E < Emax][E > Emin].min()
    ioveri = abs(peak_o/peak_r)
    
    E_o = E[i_f[E < Emax][E > Emin].idxmax()]
    E_r = E[i_f[E < Emax][E > Emin].idxmin()]
    dEp = abs(E_o-E_r)
    
    # First figure (0) of the full voltammogram
    
    plt.figure(0)
    
    plt.plot(E, i_f, color=colors[item], label=f'{item}', alpha=.7)
    
    plt.xlabel('E / V vs Ag/AgCl')
    plt.ylabel('I / $\mu$A')
    # plt.title('')
    # plt.ylim(-210,140)
    plt.legend(fontsize=10, frameon=False)
    
    # plt.savefig('20230808.png', dpi=200, bbox_inches='tight')
    
    # If blank is pertinent: figure (1) of the blank
    
    if b == 'b':
    
        plt.figure(1)
        
        plt.plot(E, i_b, color=colors[item], label=f'{item}-branco', alpha=.7)
        
        plt.xlabel('E / V vs Ag/AgCl')
        plt.ylabel('I / $\mu$A')
        # plt.title('')
        # plt.ylim(-210,140)
        plt.legend(fontsize=10, frameon=False)
        
        # plt.savefig('02082023-b.png', dpi=200, bbox_inches='tight')
    # plt.figure(1)
    
    # print(E_o)
    # print(peak_o-0.5*peak_o)
    
    # plt.plot(E, i_f, color=colors[item], label=f'{item}', alpha=1)
    # plt.axvline(x=E_o, ymin=i_f[E_o]-40, ymax=i_f[E_o]+6,linewidth=1, color='r')
    # plt.axvline(x=E_r, ymin=peak_r-0.5*peak_r, ymax=peak_r+0.5*peak_r,)
    # plt.xlabel('E / V vs Ag/AgCl')
    # plt.ylabel('I / $\mu$A')
    # plt.title('peak detection')
    
    return peak_o, peak_r, ioveri, dEp

def peaks(tech, sep, reps, scan, b, Emax, Emin):
    """
    Runs a loop of plots() functions and prints the results accordingly
    """
    
    t = stats.t.ppf(1-0.025, reps-1)
    
    if b == 'b':
        
        peaks = np.array( [plot(item, sep, tech, scan, 'b', Emax, Emin) for item in np.arange(0,reps)] ).T
    
    else: 
        
        peaks = np.array( [plot(item, sep, tech, scan, 0, Emax, Emin) for item in np.arange(0,reps)] ).T
    
    anodic = peaks[0]
    
    mean_a = anodic.mean()
    std_a = anodic.std(ddof=1)
    ci_a = std_a*t/(reps**(0.5))
    err_a = round(ci_a/mean_a*100, 0)
    
    print('PEAK ANODIC CURRENTS:')
    print('-'*40)
    print('Array with all peaks: ')
    print(anodic)
    print('Mean = ', mean_a)
    print('Std = ', std_a)
    print('CI = ', mean_a, '+-', ci_a)
    print('Error = ', err_a, '%')
    
    cathodic = peaks[1]
    
    mean_c = cathodic.mean()
    std_c = cathodic.std(ddof=1)
    ci_c = std_c*t/(reps**(0.5))
    err_c = round(ci_c/mean_c*100, 0)
    
    print('-'*40)
    print('PEAK CATHODIC CURRENTS:')
    print('-'*40)
    print('Array with all peaks: ')
    print(cathodic)
    print('Mean = ', mean_c)
    print('Std = ', std_c)
    print('CI = ', mean_c, '+-', ci_c)
    print('Error = ', err_c, '%')
    
    ioveri = peaks[2]
    
    mean_ioveri = ioveri.mean()
    std_ioveri = ioveri.std(ddof=1)
    ci_ioveri = std_ioveri*t/(reps**(0.5))
    err_ioveri = round(ci_ioveri/mean_ioveri*100, 0)
    
    print('-'*40)
    print('ANODIC/CATHODIC CURRENT RATIO:')
    print('-'*40)
    print('Array with all peaks: ')
    print(ioveri)
    print('Mean = ', mean_ioveri)
    print('Std = ', std_ioveri)
    print('CI = ', mean_ioveri, '+-', ci_ioveri)
    print('Error = ', err_ioveri, '%')
    
    dEp = peaks[3]
    
    mean_dEp = dEp.mean()
    std_dEp = dEp.std(ddof=1)
    ci_dEp = std_dEp*t/(reps**(0.5))
    err_dEp = round(ci_dEp/mean_dEp*100, 0)
    
    print('-'*40)
    print('ANODIC/CATHODIC CURRENT RATIO:')
    print('-'*40)
    print('Array with all peaks: ')
    print(dEp)
    print('Mean = ', mean_dEp)
    print('Std = ', std_dEp)
    print('CI = ', mean_dEp, '+-', ci_dEp)
    print('Error = ', err_dEp, '%')
    
    print('='*50)
    print('='*50)
    print('SUMMARY:')
            
    print('i_a:     ', round(mean_a, 4), '+-', round(ci_a, 4), 'uA', f"        ({err_a})%")
    print('i_c:     ', round(mean_c, 4), '+-', round(ci_c, 4), 'uA', f"        ({err_c})%")
    print('i_a/i_c: ', round(mean_ioveri, 4), '+-', round(ci_ioveri, 4), f"        ({err_ioveri})%")
    print('dEp:     ', round(mean_dEp,4), '+-', round(ci_dEp, 4), 'V', f"        ({err_dEp})%")
    
    print('='*50)
    print('='*50)
    
    return 

#%% OUTLIER TEST

def grubbs_stat(y):
    std_dev = np.std(y, ddof=1)
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    max_ind = np.argmax(abs_val_minus_avg)
    Gcal = max_of_deviations/ std_dev
    print("Grubbs Statistics Value : {:.4f}".format(Gcal))
    return Gcal, max_ind

def calculate_critical_value(size, alpha):
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    print("Grubbs Critical Value: {:.4f}".format(critical_value))
    return critical_value

def check_G_values(Gs, Gc, inp, max_index):
    TEST = 0
    if Gs > Gc:
        print('Position: ', max_index)
        print('{} - OUTLIER. G > G-critical: {:.4f} > {:.4f} \n'.format(inp[max_index], Gs, Gc))
        TEST = 1
    else:
        print('{}. G > G-critical: {:.4f} > {:.4f} \n'.format(inp[max_index], Gs, Gc))
        TEST = 0        
    return TEST

def ESD_Test(input_series, alpha, max_outliers):
    n = 0
    for iterations in range(max_outliers):
        Gcritical = calculate_critical_value(len(input_series), alpha)
        Gstat, max_index = grubbs_stat(input_series)
        check = check_G_values(Gstat, Gcritical, input_series, max_index)
        input_series = np.delete(input_series, max_index)
        n += check
    print(f'Found {n} outliers!')
    
    return n

def remove_outliers(data, n):
    for item in np.arange(n):
        posmax = np.argmax(data)
        data = np.delete(data, posmax)
    return data

data = pd.read_csv('testeariel.txt', sep=' ')

ox = np.array(data['ox'])
red = np.array(data['red'])
dE = np.array(data['dE'])
ioveri = ox/-red

my_dict = {'ox': ox/max(ox), 'red': red/min(red), 'dE': dE/max(dE), 'ioveri': ioveri/max(ioveri)}
fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())

n = ESD_Test(red, 0.05, 5)

# newdata = remove_outliers(data, n)

#%%
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


#%% Linear regression modelling

def poly_scale(df, degree):
    """
    Scales an dataframe for the nth {degree}
    """
    
    poly = PolynomialFeatures(degree=degree)
    scaler = MinMaxScaler()
    
    df_scaled = scaler.fit_transform(df)
    
    df_poly_scaled = poly.fit_transform(df_scaled)
    
    return df_scaled, df_poly_scaled, scaler

def reg_model(y, x, method, errors):
    """
    Fits and report a linear model ({method} = 'OLS' or 'WLS') for y(x) returning (model, coefs, r2, r2_adj)
    """
    if method == 'OLS':
    
        model = sm.OLS(y, x).fit()
        
    elif method == 'OLS':
        
        weights =  1/(err**2)
        model = sm.WLS(y, x, weights=weights).fit()
        
    coefs = np.array(model.params)    
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    
    print(model.summary())
    
    return model, coefs, r2, r2_adj

### Original data

df = pd.read_csv('dadosjou.csv')

### Copy of original data

data = df

### Parameters: xs, ys and errors

x1 = data['x']

y1 = data['y']/data['y'].max()
err = data['std']

### Creating final matrix 

X_final = pd.DataFrame([x1]).T

### Adding intercept

X_final = sm.add_constant(X_final)

### Scaling the final matrix

# X_final_scaled, X_poly_final_scaled, scaler = poly_scale(X_final, 2)

### Selecting specific polynomial terms 

# select = list([0,1])

### Actual modelling
# model_final, coefs_sm, r2, r2_adj = reg_model( y1, X_final, 'OLS')
model_final, coefs_sm, r2, r2_adj = reg_model( y1, X_final, 'WLS', err)


### Graphing linear regression

x = np.linspace(0, 5000, 400)
yy = coefs_sm[0] + coefs_sm[1]*x

plt.scatter(x1,y1, color='black')
plt.plot(x, yy, color='red')
# plt.title('{} Hz / {} mV'.format(data['freq'][number], data['amp'][number]))

plt.ylabel('Sinal / cps')
# plt.ylim(-210,140)
plt.xlabel('Concentração / ppb')




#%%


peaks = np.array([plot(item) for item in [0,1,4]])

#%%

peaks = np.array([plot(item) for item in [2,3,5]])

#%%

for i in np.arange(500):
    
    print(stats.t.ppf(1-0.025, i-1)/(i**0.5))
    
#%% PCA modelling

#Load dataset
data = pd.read_csv('Cu.csv')
data.set_index('0', inplace=True)

#Centering
scaled_data = preprocessing.scale(data)

pca = PCA()
# pca.fit(scaled_data)
pca_data = pca.fit_transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(pca_data, index=list(data.index), columns=labels)
 
plt.scatter(pca_df.PC1, pca_df.PC2, color='#343E3D')

plt.axhline(0, -100, 100, color='#FF5252', alpha=.5)
plt.axvline(0, -100, 100, color='#FF5252', alpha=.5)

plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
 
for sample in pca_df.index:
    for i in range(3):

        plt.annotate(sample, (pca_df.PC1.loc[sample][i], pca_df.PC2.loc[sample][i]), fontsize=9)

 
plt.show()
#%%
 ## 6 cores ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF']

 ## 6 cores ['preto', 'vermelho', 'amarelo', 'verde', 'azul', 'rosa']

#########################
#
# Determine which genes had the biggest influence on PC1
#
#########################
 
## get the name of the top 10 measurements (genes) that contribute
## most to pc1.
## first, get the loading scores
loading_scores = pd.Series(pca.components_[0], index=np.arange(0,1000))
## now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
 
# get the names of the top 10 genes
top_10_genes = sorted_loading_scores[0:100].index.values
 
## print the gene names and their scores (and +/- sign)
print(loading_scores[top_10_genes])

#%%

import umap

reducer = umap.UMAP(5)

embedding = reducer.fit_transform(pca_data)

#%%

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=['red', 'red', 'red', 'blue', 'blue', 'blue', 'yellow', 'yellow', 'yellow'])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Penguin dataset', fontsize=24);