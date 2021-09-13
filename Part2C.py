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
import time
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


# In[3]:


def make_batch(X,Y,batch_size):

    data_join = np.hstack((X,Y))
    
    # Shuffle the data
    np.random.shuffle(data_join)
    
    #print(data_join)
    
    # Find the number of batch possible
    batch_n = data_join.shape[0]//batch_size
    
    # Store each batch
    batch = []
    
    i=0
    for i in range(batch_n):
        tmp = data_join[i * batch_size : (i+1) * batch_size, :]
        X_new = tmp[:,[0,1,2]]
        Y_new = tmp[:,[3]]
        batch.append((X_new,Y_new))
        
    # If data_join is not proper multiple of batch_size then make a batch of the remaining data
    
    if(data_join.shape[0]%batch_size!=0):
        tmp = data_join[(i+1) * batch_size : , :]
        X_new = tmp[:,[0,1,2]]
        Y_new = tmp[:,[3]]
        batch.append((X_new,Y_new))
    
    return batch


# In[4]:


def find_hypothesis(X,theta):
    return np.dot(X,theta)

def find_J_theta(X,Y,theta):
    hypothesis = find_hypothesis(X,theta)
    difference =  Y - hypothesis
    sum_sq = np.dot(difference.transpose(),difference)
    return sum_sq[0,0]/(2*X.shape[0])

def gradient(X,Y,theta):
    term1 = np.dot(X.T,X)
    term2 = np.dot(term1,theta)
    term3 = np.dot(X.transpose(),Y)
    return(term2-term3)
    


def batch_SGD(X,Y, learning_rate=0.000001, batch_size=10000, error_threshold = 1e-4):
    
    # theta shape is number of unknown parameters * 1 (2*1)
    # print(X.shape)
    theta = np.zeros((X.shape[1],1)) 
    
    list_error = []
    list_theta = []
    
    list_theta.append(theta)
    
    # Finding the error
    J_theta = find_J_theta(X,Y,theta)
    list_error.append(J_theta)
    
    epoch = 0
    while(True):
        epoch+=1
        batches =  make_batch(X,Y,batch_size)
        
        for X_batch, Y_batch  in batches:
            theta = theta - learning_rate * gradient(X_batch, Y_batch, theta)  
        
        J_theta_new = find_J_theta(X,Y,theta)
            
        list_error.append(J_theta_new)
        list_theta.append(theta)
        
        if(abs(J_theta_new-J_theta)<error_threshold or epoch>20):
            return theta,list_error, list_theta,epoch
        J_theta = J_theta_new.copy()   
            


# In[5]:


# Since we know the expected theta thus we can find the Y = theta * X
Y = np.matmul(X,theta)
Y = Y.reshape((-1,1))


# In[7]:


batch_size = [1, 100, 10000, 1000000]
learning_rate = [0.001, 0.0005, 0.000001, 0.000001]
threshold = [1e-5, 1e-5, 1e-2, 1e-2]

# Read the data
X_tmp = genfromtxt('data/q2/q2test.csv', delimiter=',')

# First row of this data is nan remove that row
X_tmp = X_tmp[1:]

X_test = []
Y_test = []

for i in range(len(X_tmp)):
    X_test.append([1.,X_tmp[i][0],X_tmp[i][1]])
    Y_test.append([X_tmp[i][2]])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# error for the given theta
# Initializing theta
theta1 = np.array([3,1,2]).reshape((-1,1))

print("Error for the original theta :",find_J_theta(X_test,Y_test,theta1))

for i in range(len(batch_size)):
    print("For batch_size :",batch_size[i],", learning_rate :",learning_rate[i],", threshold :",threshold[i])
    
    theta, lst_error,lst_theta,epoch = batch_SGD(X, Y,learning_rate[i],batch_size[i],threshold[i])
    print("Number of epochs :",epoch)
    print("Final Theta :\n",theta)
    print("Error :",find_J_theta(X_test,Y_test,theta))


# In[ ]:





# In[ ]:





# In[ ]:




