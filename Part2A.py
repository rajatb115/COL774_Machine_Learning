#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os
import sys
from scipy.stats import norm
from sklearn import preprocessing
import math
debug = True
from numpy import genfromtxt


# In[2]:


sample_size = 1000000

# Normal Distribution (Data)
X1 = np.random.normal(3, 4, sample_size)
X2 = np.random.normal(-1, 4, sample_size)

X = []
# Adding the intercept
for i in range(sample_size):
    X.append([1.0,X1[i],X2[i]])

X= np.array(X)

# noise
e = np.random.normal(0,math.sqrt(2))

# Initializing theta
theta = np.array([3,1,2]).reshape((-1,1))


# In[ ]:




