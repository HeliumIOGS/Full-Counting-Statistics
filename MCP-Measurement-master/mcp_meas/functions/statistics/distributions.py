# -*- coding: utf-8 -*-
"""
Author   : alex
Created  : 2021-05-18 14:13:58
Modified : 2021-05-18 14:20:11

Comments : Theoretical population distributions
"""

# %% IMPORTS

# -- global

import numpy as np
from scipy.special import factorial


# %% FUNCTIONS

# == theoretical distributions

# -- Poisson
@np.vectorize
def poisson(n, a=1):
    return a ** n / factorial(n) * np.exp(-a)


# -- Thermal / Bose
@np.vectorize
def thermal(n, n_mean=2):
    q = n_mean / (1 + n_mean)
    return (1 - q) * q ** n


# -- Multi thermal
@np.vectorize
def multi_thermal(n, M, n_mean):
    A = factorial(n + M - 1) / factorial(n) / factorial(M - 1)
    B = (n_mean / M) ** n / (1 + n_mean / M) ** (n + M)
    return A * B