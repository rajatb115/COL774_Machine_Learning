#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os
import sys
from scipy.stats import norm
from sklearn import preprocessing

debug = False


# In[4]:


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


# In[5]:


X = genfromtxt('data/q4/q4x.dat')
Y = genfromtxt('data/q4/q4y.dat',dtype='U')

Y_normal = []

for i in Y:
    if(i=='Alaska'):
        Y_normal.append([int(1)])
    else:
        Y_normal.append([int(0)])
Y = np.array(Y)
X = np.array(X)     

X_normal = normalize(X)
X_normal = np.array(X_normal).reshape((-1,2,1))
Y_normal = np.array(Y_normal).reshape((-1,1))

tmp1 = 0
tmp2 = 0

tmp3 = np.zeros((2,1))
tmp4 = np.zeros((2,1))

for i in range(X_normal.shape[0]):

    if(Y_normal[i][0]==0):
        tmp1+=1
        tmp3+=X_normal[i]
        #for j in range(2):
        #    tmp3[j] += X_normal[i][j]
    else:
        tmp2+=1
        tmp4+=X_normal[i]
        #for j in range(2):
        #    tmp4[j] += X_normal[i][j]

mew_0 = tmp3/tmp1
mew_1 = tmp4/tmp2
mew = np.array([mew_0,mew_1])

# # of times y(i) = 1 in the data / # of examples
phi = tmp2/X_normal.shape[0]

etha = np.zeros((2,2))

for i in range(X_normal.shape[0]):
    etha += np.dot(X_normal[i] - mew[Y_normal[i][0]], (X_normal[i] - mew[Y_normal[i][0]]).transpose() )
etha = etha/X_normal.shape[0]

# print("Phi :",phi)
# print("\nmew0 :\n",mew_0)
# print("\nmew1 :\n",mew_1)
# print("\nCovariance matrix :\n",etha)


# In[5]:


X1 = []
Y1 = []

X2 = []
Y2 = []

for i in range(Y_normal.shape[0]):
    if(Y_normal[i]==0):
        X1.append(X_normal[i][0])
        Y1.append(X_normal[i][1])
    else:
        X2.append(X_normal[i][0])
        Y2.append(X_normal[i][1])
        
c = np.log(phi/(1-phi))
term1 = 0.5 * (np.dot(np.dot(mew[0].T,np.linalg.inv(etha)),mew[0]) - np.dot(np.dot(mew[1].T,np.linalg.inv(etha)),mew[1]))
term2 = np.dot((mew[0] - mew[1]).transpose(),np.linalg.inv(etha))


X3 = []
Y3 = []

# term2[0]*X[0] + term2[1]*X[1] = term1  - c
for i in range(X.shape[0]):
    X3.append(X_normal[i][0])
    k=((term1 -c - term2[0][0]*X_normal[i][0])/term2[0][1])
    Y3.append(k[0][0])


# In[7]:


c = np.log(phi/(1-phi))
term1 = 0.5 * (np.dot(np.dot(mew[0].T,np.linalg.inv(etha)),mew[0]) - np.dot(np.dot(mew[1].T,np.linalg.inv(etha)),mew[1]))
term2 = np.dot((mew[0] - mew[1]).transpose(),np.linalg.inv(etha))


X3 = []
Y3 = []

# term2[0]*X[0] + term2[1]*X[1] = term1  - c
for i in range(X.shape[0]):
    X3.append(X_normal[i][0])
    k=((term1 -c - term2[0][0]*X_normal[i][0])/term2[0][1])
    Y3.append(k[0][0])



# # of times y(i) = 1 in the data / # of examples
n_phi = tmp2/X_normal.shape[0]

n_mew_0 = tmp3/tmp1
n_mew_1 = tmp4/tmp2
n_mew = np.array([mew_0,mew_1])

etha0 = np.zeros((2,2))

for i in range(X.shape[0]):
    if(Y_normal[i][0]==0):
        etha0 += (X_normal[i]-n_mew[0])*(X_normal[i]-n_mew[0]).transpose()
etha0 = etha0/tmp1

etha1 = np.zeros((2,2))

for i in range(X.shape[0]):
    if(Y_normal[i][0]==1):
        etha1 += (X_normal[i]-n_mew[1])*(X_normal[i]-n_mew[1]).transpose()
etha1 = etha1/tmp2


print("Phi :",n_phi)
print("\nMew :\n",n_mew)
print("\ncovariance matrix 0 :\n",etha0)
print("\ncovariance matrix 1 :\n",etha1)


# In[5]:


c1 = np.log(n_phi/(1-n_phi)) + 0.5*(np.log(np.linalg.det(etha0)/np.linalg.det(etha1)))

#c1 = np.dot(np.dot(n_mew[1].transpose(),np.linalg.inv(etha1)),n_mew[1]) - np.dot(np.dot(n_mew[0].transpose(),np.linalg.inv(etha0)),n_mew[0])
etha0_inv = np.linalg.pinv(etha0)
etha1_inv = np.linalg.pinv(etha1)

# choose some points
X_n = np.linspace(np.max(X_normal[:,0]),np.min(X_normal[:,0]),20)
Y_n = np.linspace(np.max(X_normal[:,1]),np.min(X_normal[:,1]),20)


n_mew_0 = np.array(n_mew_0).reshape((1,-1))
n_mew_1 = np.array(n_mew_1).reshape((1,-1))

X_n, Y_n = np.meshgrid(X_n,Y_n)
Z_n = np.zeros((X_n.shape[0],Y_n.shape[0]))

for i in range(X_n.shape[0]):
    for j in range(Y_n.shape[0]):
        tmp_n = np.array([X_n[i][j], Y_n[i][j]]) 
        val = 0.5 * (np.dot(np.dot((tmp_n-n_mew_0[0]).T , etha0_inv), (tmp_n-n_mew_0[0])) - np.dot(np.dot((tmp_n-n_mew_1[0]).T , etha1_inv), (tmp_n-n_mew_1[0])))
        Z_n[i][j] = val + c1


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(X1, Y1, label = 'Canada',color="blue")
plt.scatter(X2, Y2, label ='Alaska',color="green")
plt.plot(X3,Y3)
plt.contour(X_n,Y_n,Z_n,[0])
plt.legend(loc='upper right')
plt.xlabel("Ring diameter in fresh water")
plt.ylabel("Ring diameter in marine water")
plt.title("Gaussian Discrmimant Analysis - Quadratic plot")
plt.savefig('output/Ques4(E).png')
plt.show()

