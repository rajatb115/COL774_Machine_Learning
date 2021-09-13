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

debug = False


# In[2]:


def normalize(x):
    
    X1 = []
    X2 = []
    for i in x:
        X1.append(i[0])
        X2.append(i[1])
    
    mean = np.mean(X1)
    std = np.std(X1)
    
    X1 = X1-mean
    X1=X1/std
    
    mean = np.mean(X2)
    std = np.std(X2)
        
    X2 = X2-mean
    X2=X2/std
    
    x=[]
    
    for i in range(len(X2)):
        x.append([X1[i],X2[i]])
    
    return x
        


# In[3]:


def g(theta,x):
    hyp = np.dot(theta.transpose(),x)
    return 1 / (1 + np.exp(-1*hyp))

# log likelyhood of theta
def log_l(x,y,theta):
    sm = 0
    
    for i in range(len(y)):
        sm += y[i] * np.log(g(theta, x[i]))
        sm +=(1 - y[i])*np.log(1 - g(theta, x[i]))
    return sm

# first order derivative of log likelyhood over all the dim
def d_log_l(x,y,theta):
    sm = np.zeros((x.shape[1],1))
    
    for i in range(x.shape[1]): # this is dimentional space
        for j in range(x.shape[0]): # number of items
            sm[i] += (y[j]-g(theta,x[j]) )*x[j][i]
    
    return sm

# second order derivative of log likelyhood over all the dimentions    
def dd_log_l(x,y,theta):
    
    # hessian is the partial derivative over two dimentions
    # hessian is symmetric matrix hessian[i][j] = hessian[j][i]
    sm = np.zeros((x.shape[1],x.shape[1]))
    
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            for k in range(x.shape[0]):
                sm[i][j] += g(theta,x[k])*(1-g(theta,x[k]))*x[k][i]*x[k][j]
    
    return sm


def newton(x,y,threshold,learning_rate):
    theta = np.zeros((x.shape[1],1))
    
    error_p = log_l(x,y,theta)
    error_lis = []
    error_lis.append(error_p)
    
    epoch = 0
    while(True):
        epoch+=1
        
        theta = theta + learning_rate*np.matmul((np.linalg.inv(dd_log_l(x,y,theta))), d_log_l(x,y,theta))
        error = log_l(x,y,theta)
        error_lis.append(error)
        if(abs(error_p - error) < threshold or epoch > 30):
            return theta,error_lis,epoch
        error_p = error


# In[7]:


X = genfromtxt('data/q3/logisticX.csv', delimiter=',')
Y = genfromtxt('data/q3/logisticY.csv',delimiter=',')

X_normal = normalize(X)

# Add intercept
X_normal = np.hstack((np.ones((len(X), 1)), X_normal))

X_normal = np.array(X_normal).reshape((-1,3))
Y = np.array(Y).reshape((-1,1))

# print(X_normal.shape)
# print(Y.shape)

theta_f,error_list,epochs = newton(X_normal,Y,1e-8,1.)

print("Theta :\n",theta_f)
print("Epochs :",epochs)


# In[ ]:




