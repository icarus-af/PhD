from patsy import dmatrices

import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn import preprocessing

from scipy.signal import find_peaks

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

colors = ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF' , '#FF5252']
black, red, yellow, green, blue, pink = '#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF'

## 6 cores ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF']

## 6 cores ['preto', 'vermelho', 'amarelo', 'verde', 'azul', 'rosa']

#%%

def getcm(n, cmap):
    """
    Get a sequence of n colors from a specified colormap.
    
    Parameters:
    - n: Number of colors to retrieve from the colormap.
    - cmap: Name of the colormap to use (e.g., 'viridis', 'RedBu', etc.).
    
    Returns:
    - colors: List of n colors retrieved from the specified colormap.
    """   
    if n == 6:

        colors = ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF']

    else:
        
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    
    return colors

# def plot(item, colors=['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF'], Emin=-100000, Emax=100000, b=None, scan=1, sep=',', tech='cv'):
#     """
#     Reads an {item}.txt file filtered by {scan} column and outputs {peak_o, peak_r, ioveri, dEp}
#     Different conditions based on {tech} = 'cv' or 'swv'
#     if b           = 'b' then blank is considered on analysis
#     Emax and Emin  = potential ranges for peak detection
#     """
    
#     # Read data for Cyclic Voltammetry analysis
#     if tech == 'cv':
    
#         df = pd.read_csv(f'{item}-a.txt', sep=sep)
        
#         if 'Scan' in df.columns:
        
#             df = df[df['Scan'] == scan]
        
#         E = df['WE(1).Potential (V)']
#         i = df['WE(1).Current (A)']*1e6
#         i_f = i
          
#         if b is not None:
            
#             df_b = pd.read_csv(f"{item}-b.txt", sep=sep)
            
#             if 'Scan' in df_b.columns:
            
#                 df_b = df_b[df_b['Scan'] == scan]
                
#             i_b = df_b['WE(1).Current (A)']*1e6
#             i_f = i - i_b
            
#     # Read data for Squared Wave Voltammetry analysis
    
#     elif tech == 'swv':
        
#         df = pd.read_csv(f'{item}-a.txt', sep=sep)
        
#         if 'Scan' in df.columns:
        
#             df = df[df['Scan'] == scan]
        
#         E = df['Potential applied (V)']
#         i = df['WE(1).δ.Current (A)']*1e6
#         i_f = i
          
#         if b is not None:
            
#             df_b = pd.read_csv(f'{item}-b.txt', sep=sep)
            
#             if 'Scan' in df_b.columns:
            
#                 df_b = df_b[df_b['Scan'] == scan]
                
#             i_b = df_b['WE(1).δ.Current (A)']*1e6
#             i_f = i - i_b
            
#     # Calculate pertinent data analysis
           
#     peak_o = i_f[E < Emax][E > Emin].max()
#     peak_r = i_f[E < Emax][E > Emin].min()
#     ioveri = abs(peak_o/peak_r)
    
#     E_o = E[i_f[E < Emax][E > Emin].idxmax()]
#     E_r = E[i_f[E < Emax][E > Emin].idxmin()]
#     dEp = abs(E_o-E_r)
    
#     # First figure (0) of the full voltammogram
    
#     plt.figure(0)
    
#     plt.plot(E, i_f, color=colors[item], label=f'{item}', alpha=.7)
    
#     plt.xlabel('E / V vs Ag/AgCl')
#     plt.ylabel('I / $\mu$A')
#     # plt.title('')
#     # plt.ylim(-210,140)
#     plt.legend(fontsize=10, frameon=False)
    
#     # plt.savefig('20230808.png', dpi=200, bbox_inches='tight')
    
#     # If blank is pertinent: figure (1) of the blank
    
#     if b is not None:
    
#         plt.figure(1)
        
#         plt.plot(E, i_b, color=colors[item], label=f'{item}-branco', alpha=.7)
        
#         plt.xlabel('E / V vs Ag/AgCl')
#         plt.ylabel('I / $\mu$A')
#         # plt.title('')
#         # plt.ylim(-210,140)
#         plt.legend(fontsize=10, frameon=False)
        
#         # plt.savefig('02082023-b.png', dpi=200, bbox_inches='tight')
#     plt.figure(2)
    
#     print('PEAKPEAK')
#     print(E_o)
#     print('-'*90)
#     print(peak_o)
#     print('-'*90)
#     peaks, _ = find_peaks(i_f, prominence=1)
#     print(peaks)
#     print('-'*90)
    
#     plt.plot(E, i_f, color=colors[item], label=f'{item}', alpha=1)
#     plt.axvline(x=E[peaks[0]], linewidth=2, color='red')
    
#     ### plt.axvline(x=E_r, ymin=peak_r-0.5*peak_r, ymax=peak_r+0.5*peak_r,) ###
    
#     plt.xlabel('E / V vs Ag/AgCl')
#     plt.ylabel('I / $\mu$A')
#     plt.title('peak detection')
    
#     return peak_o, peak_r, ioveri, dEp


def plot(item, colors=['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF'], Emin=-100000, Emax=100000, b=None, scan=1, sep=',', tech='cv', area=1):
    """
    Reads a data file from defined voltammetric {tech} ('cv' for Cyclic Voltammetry or 'swv' for Squared Wave Voltammetry) corresponding to the {item} and filters it based on the {scan} column.
    Calculates and plots relevant peak information based on specified conditions:
    
    - Different colors can be specified for plotting different items.
    - Emin and Emax define the potential range for peak detection.
    - If {b} is specified, it considers a blank for analysis.
    - {scan} specifies the scan column to be used.
    - {sep} defines the separator used in the data file.
    - {area} represents the area for current density calculation.
    
    Outputs:
    - peak_o: Maximum current peak within the specified potential range.
    - peak_r: Minimum current peak within the specified potential range.
    - ioveri: Absolute ratio of peak_o to peak_r.
    - dEp: Absolute potential difference between peak_o and peak_r.
    
    Parameters:
    - item: String representing the item name or identifier.
    - colors: List of color codes for plotting different items (default color list provided).
    - Emin: Minimum potential for peak detection (default: -100000).
    - Emax: Maximum potential for peak detection (default: 100000).
    - b: Specifies whether a blank should be considered for analysis (default: None).
    - scan: Specifies the scan column to be used (default: 1).
    - sep: Separator used in the data file (default: ',').
    - tech: Specifies the type of voltammetry analysis ('cv' for Cyclic Voltammetry or 'swv' for Squared Wave Voltammetry, default: 'cv').
    - area: Area for current density calculation (default: 1).

    Returns:
    Tuple containing peak_o, peak_r, ioveri, and dEp.
    """
    
    # Read data for Cyclic Voltammetry analysis
    if tech == 'cv':
    
        df = pd.read_csv(f'{item}-a.txt', sep=sep)
        
        if 'Scan' in df.columns:
        
            df = df[df['Scan'] == scan]
        
        E = df['WE(1).Potential (V)']
        i = df['WE(1).Current (A)']*1e6
        i_f = i/area
          
        if b is not None:
            
            df_b = pd.read_csv(f"{item}-b.txt", sep=sep)
            
            if 'Scan' in df_b.columns:
            
                df_b = df_b[df_b['Scan'] == scan]
                
            i_b = df_b['WE(1).Current (A)']*1e6/area
            i_f = i - i_b
            
    # Read data for Squared Wave Voltammetry analysis
    
    elif tech == 'swv':
        
        df = pd.read_csv(f'{item}-a.txt', sep=sep)
        
        if 'Scan' in df.columns:
        
            df = df[df['Scan'] == scan]
        
        E = df['Potential applied (V)']
        i = df['WE(1).δ.Current (A)']*1e6
        i_f = i/area
          
        if b is not None:
            
            df_b = pd.read_csv(f'{item}-b.txt', sep=sep)
            
            if 'Scan' in df_b.columns:
            
                df_b = df_b[df_b['Scan'] == scan]
                
            i_b = df_b['WE(1).δ.Current (A)']*1e6/area
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
    plt.vlines(E_o, peak_o-0.05*peak_o, peak_o+0.05*peak_o, color=black)
    plt.vlines(E_r, peak_r-0.05*peak_r, peak_r+0.05*peak_r, color=black)
    
    plt.xlabel('E / V vs Ag/AgCl')
    plt.ylabel('I / $\mu$A')
    
    if area != 1:
        plt.ylabel('J / $\mu$A $mm^{-2}$')
    # plt.title('')
    # plt.ylim(-210,140)
    plt.legend(fontsize=10, frameon=False)
    
    # plt.savefig('20230808.png', dpi=200, bbox_inches='tight')
    
    # If blank is pertinent: figure (1) of the blank
    
    if b is not None:
    
        plt.figure(1)
        
        plt.plot(E, i_b, color=colors[item], label=f'{item}-branco', alpha=.7)
        
        plt.xlabel('E / V vs Ag/AgCl')
        plt.ylabel('I / $\mu$A')
        if area != 1:
            plt.ylabel('J / $\mu$A $mm^{-2}$')
        # plt.title('')
        # plt.ylim(-210,140)
        plt.legend(fontsize=10, frameon=False)
        
        # plt.savefig('02082023-b.png', dpi=200, bbox_inches='tight')
    plt.figure(2)
    
    print('PEAKPEAK')
    print(E_o)
    print('-'*90)
    print(peak_o)
    print('-'*90)
    peaks, _ = find_peaks(i_f, prominence=1)
    print(peaks)
    print('-'*90)
    
    plt.plot(E, i_f, color=colors[item], label=f'{item}', alpha=1)
    # plt.axvline(x=E[peaks[0]], linewidth=2, color='red')
    
    # plt.axvline(x=E_r, ymin=peak_r-0.5*peak_r, ymax=peak_r+0.5*peak_r,) ###
    
    plt.xlabel('E / V vs Ag/AgCl')
    plt.ylabel('I / $\mu$A')
    if area != 1:
        plt.ylabel('J / $\mu$A $mm^{-2}$')
    # plt.title('peak detection')
    
    return peak_o, peak_r, ioveri, dEp


#%%
def peaks(reps, cmap, Emin=-100000, Emax=100000, b=None, scan=1, sep=',', tech='cv', area=1):
    """
    Runs a loop of the plot() function for multiple items and prints statistical results.
    
    Parameters:
    - reps: Number of repetitions or items to be analyzed.
    - cmap: Colormap for determining colors (used in getcm() function).
    - Emin: Minimum potential for peak detection (default: -100000).
    - Emax: Maximum potential for peak detection (default: 100000).
    - b: Specifies whether a blank should be considered for analysis (default: None).
    - scan: Specifies the scan column to be used (default: 1).
    - sep: Separator used in the data file (default: ',').
    - tech: Specifies the type of voltammetry analysis ('cv' for Cyclic Voltammetry or 'swv' for Squared Wave Voltammetry, default: 'cv').
    - area: Area for current density calculation (default: 1).

    Returns:
    - peaks: NumPy array containing peak information for each repetition.
    
    Outputs:
    - Statistical results including mean, standard deviation, confidence interval, and error percentage for:
      - Anodic peak currents (i_a).
      - Cathodic peak currents (i_c).
      - Anodic/Cathodic current ratio (i_a/i_c).
      - Absolute potential difference between peaks (dEp).
    """
    
    colors = getcm(reps, cmap)
    
    t = stats.t.ppf(1-0.025, reps-1)
    
    if b is not None:
        
        peaks = np.array( [plot(item, colors=colors, Emin=Emin, Emax=Emax, tech=tech, sep=sep, scan=scan, b='b', area=area) for item in np.arange(0,reps)] ).T
    
    else: 
        
        peaks = np.array( [plot(item, colors=colors, Emin=Emin, Emax=Emax, tech=tech, sep=sep, scan=scan, area=area) for item in np.arange(0,reps)] ).T
    
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
    
    return peaks

#%% OUTLIER TEST

def grubbs_stat(y):
    """
    Calculate the Grubbs statistics value for outlier detection.

    Parameters:
    - y: Input data (array-like) for which outliers are to be detected.

    Returns:
    - Gcal: Grubbs statistics value.
    - max_ind: Index of the maximum deviation, used for outlier detection.
    """
    
    std_dev = np.std(y, ddof=1)
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    max_ind = np.argmax(abs_val_minus_avg)
    Gcal = max_of_deviations/ std_dev
    print("Grubbs Statistics Value : {:.4f}".format(Gcal))
    return Gcal, max_ind

def calculate_critical_value(size, alpha):
    """
   Calculate the critical value for the Grubbs test.

   Parameters:
   - size: Size of the sample.
   - alpha: Significance level (e.g., 0.05 for a 95% confidence level).

   Returns:
   - critical_value: Critical value for outlier detection using the Grubbs test.
   """
   
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    print("Grubbs Critical Value: {:.4f}".format(critical_value))
    return critical_value

def check_G_values(Gs, Gc, inp, max_index):
    """
    Check if the Grubbs statistics value exceeds the critical value.
    
    Parameters:
    - Gs: Grubbs statistics value.
    - Gc: Critical value for the Grubbs test.
    - inp: Input data array.
    - max_index: Index of the maximum deviation in the input data.
    
    Returns:
    - TEST: Flag indicating if the data point is an outlier (1 for outlier, 0 otherwise).
    """
    
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
    """
    Apply the Extreme Studentized Deviate (ESD) test to detect and remove outliers.
    
    Parameters:
    - input_series: Input data array for outlier detection.
    - alpha: Significance level (e.g., 0.05 for a 95% confidence level).
    - max_outliers: Maximum number of outliers to detect and remove.
    
    Returns:
    - n: Number of outliers detected and removed.
    """
    
    n = 0
    for iterations in range(max_outliers):
        Gcritical = calculate_critical_value(len(input_series), alpha)
        Gstat, max_index = grubbs_stat(input_series)
        check = check_G_values(Gstat, Gcritical, input_series, max_index)
        input_series = np.delete(input_series, max_index)
        n += check
    print(f'Found {n} outliers!')
    
    return n


# NOT GENERALIZED FUNCTION
# def remove_outliers(data, n):
#     """
#     Remove outliers from the input data.
    
#     Parameters:
#     - data: Input data array.
#     - n: Number of outliers to remove.
    
#     Returns:
#     - data: Input data with outliers removed.
#     """
#     for item in np.arange(n):
#         posmax = np.argmax(data)
#         data = np.delete(data, posmax)
#     return data

def boxplots(my_dict):
    
    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    
    return fig
