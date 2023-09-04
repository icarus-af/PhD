# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:45:07 2023

@author: lesql
"""

import numpy as np

def mean_propagation(array):
    
    array = np.array(array)
    
    mean = np.mean(array)
    std = np.std(array, ddof=1)
    
    return mean, std

a = [2.0, 2.1, 2.3, 2.4, 2.1, 2.3, 2.3, 1.9, 2.2, 2.0]
b = [2.8, 3.0, 3.0, 3.1, 3.1, 2.8, 3.2, 3.2, 2.9, 3.1]



mean_propagation(a)
# mean_propagation(b)