#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:25:11 2020

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from scipy.stats import multivariate_normal

S=2**2

def grad(w,X,t,S):
   '''
    Computes the gradient of the energy function

    Parameters
    ----------
    w : array_like
        evaluation point
    X : array_like
        matrix with the given vectors x as columns
    t : array_like
        vector of labels (+/- 1)
    S : float
        assumed value of the ws' variance

    Returns
    -------
    g : array_like
        the gradient

    '''
   g=np.ndarray(len(w))
   wt=np.transpose(w)
   g[1:]=w[1:]/S+np.sum(-X[1:]*t*np.exp(-t*np.dot(wt,X))/(1+np.exp(-t*np.dot(wt,X))),axis=1)
   g[0]=np.sum(np.exp(-t*np.dot(wt,X))*-t/(1+np.exp(-t*np.dot(wt,X))))
   return g

def nagm(K,X,t,S):
    '''
    Nesterov Accelerated Gradient Method

    Parameters
    ----------
    K : int
        number of steps to be performed
    X : array_like
        matrix with the given vectors x as columns
    t : array_like
        vector of labels (+/- 1)
    S : float
        assumed value of the ws' variance

    Returns
    -------
    w2 : array_like
        minimizer

    '''
    N=len(X)
    w1=np.random.rand(N)
    w2=w1
    L=np.zeros([len(X),len(X)])
    for i in range(1,len(X[0])):
        L=L+np.dot(np.transpose(X[:,i]),X[:,i])
    L=np.linalg.svd(L)[1]
    L=max(L)
    L=L/4
    L=L+1/S
    for k in range(1,K+1):
        b=(k-1)/(k+1)
        v=w2+b*(w2-w1)
        w1=w2
        w2=v-1/L*grad(v,X,t,S)
        #w2=v-1/L*approx_fprime(v, E, 1e-6, X, t)
    return w2

def E(w,X,t,S):
    '''
    energy function

    Parameters
    ----------
    w : array_like
        weight vector
    X : array_like
        matrix with the given vectors x as columns
    t : array_like
        vector of labels (+/- 1)
    S : float
        assumed value of the ws' variance

    Returns
    -------
    ret : float
        energy

    '''
    ret=multivariate_normal.pdf(w[1:],mean=np.zeros(len(w)-1),cov=np.diag([S]*(len(w)-1)))
    ret=-np.log(ret)
    ret+=np.sum(np.log(1+np.exp((-t*np.dot(np.transpose(w),X)))))
    return ret

def sigmoid(x):
    return 1/(1+np.exp(-x))

def p(x,w):
    xt=np.transpose(x)
    return np.transpose(sigmoid(w[0]+np.dot(xt,w[1:])))

#generate the sample points
data1=np.random.multivariate_normal([6.5,2], [[0.8, 0],[0,0.7]],500)
data2=np.random.multivariate_normal([0.5,2], [[1, 0],[0,0.2]],500)
#generate the matrix
X=np.concatenate((np.transpose(data1),np.transpose(data2)),axis=1)
X=np.concatenate((np.ones((1,len(X[0]))),X))
#generate the labels
t=np.concatenate((np.ones(500),-1*np.ones(500)))

plt.scatter(data1[:,0],data1[:,1],edgecolors='blue',label='$\mu_1=(6.5,2); \Sigma_1$')
plt.scatter(data2[:,0],data2[:,1],edgecolors='red',label='$\mu_2=(0.5,2); \Sigma_2$')
plt.savefig('tex/points.pdf')
plt.show()

#compute optimal weight vector
S_list=[10**k for k in range(-4,5)]
for k in range(len(S_list)):
    S=S_list[k]
    #compare our results to the scipy's approximation
    for i in range(10):
        w=np.random.randint(1,10,3)
        print(approx_fprime(w, E, 1e-6, X, t,S))
        print(grad(w,X,t,S))
        print(approx_fprime(w, E, 1e-6, X, t,S)/grad(w,X,t,S))#all entries should be close to 1
        print('-'*43)
    #The values are very similar
    
    w_star=nagm(1000,X,t,S)
    xs=np.linspace(min(X[1]),max(X[1]))
    ys=(w_star[0]+w_star[1]*xs)/-w_star[2]
    
    MG = np.meshgrid(xs,ys)
    
    ZS=p(MG,w_star)
    plt.contourf(xs,ys,np.transpose(ZS),levels=np.linspace(0,1,20))
    plt.colorbar()
    
    plt.scatter(data1[:,0],data1[:,1],edgecolors='blue',label='$\mu_1=(6.5,2); \Sigma_1$')
    plt.scatter(data2[:,0],data2[:,1],edgecolors='red',label='$\mu_2=(0.5,2); \Sigma_2$')
    plt.plot(xs,ys,c='orange',label='decision boundary')
    plt.legend()
    plt.savefig('tex/simulated-'+str(k)+'.pdf')
    plt.show()
    print('#'*43)

##############################################################################
spam_train=np.load('spam_train.npy')
spam_train=np.transpose(spam_train)

labels=spam_train[-1]
spam_train=spam_train[:-1]

spam_val=np.load('spam_val.npy')
spam_val=np.transpose(spam_val)

val_labels=spam_val[-1]
spam_val=spam_val[:-1]

X=np.concatenate((np.ones((1,len(spam_train[0]))),spam_train))
print('S^2\t\t training\t\t validation')
print('_'*66)
for k in range(len(S_list)):
    S=S_list[k]
    w_star=nagm(1000,X,labels,S)
    
    train_predictions=2*(p(spam_train,w_star)>=0.5)-1
    A_train=1/len(spam_train[0])*sum(train_predictions==labels)#60%, 84%, 90%, 91% for S>=1e-1
    
    val_predictions=2*(p(spam_val,w_star)>=0.5)-1
    A_val=1/len(spam_val[0])*sum(val_predictions==val_labels)#60%, 84%, 88%, 91% for S>=1e-1
    print(str(S)+'\t\t'+ str(A_train)+'\t\t'+str(A_val))
