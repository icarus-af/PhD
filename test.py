# -*- coding: utf-8 -*-


import sys
sys.path.insert(0, 'G:\My Drive\Atual (1)\PESQUISA\GitRepos\PhD')
sys.path.insert(0, 'G:\Meu Drive\Atual (1)\PESQUISA\GitRepos\PhD')

import matplotlib.pyplot as plt
import numpy as np

import data_treatment as dt

#%% CHOOSE COLORS


#%%
peaks = dt.peaks('cv', '	', 9, 2, 'f', .9, -.4, 'Blues_r')
