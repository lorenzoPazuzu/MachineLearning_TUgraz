#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:37:23 2020

"""

import numpy as np
import matplotlib.pyplot as plt

#making sure the data generated is always the same
#np.random.seed(2)
N=100

x = []
t = []
eta = []
risk_array = []
pi = np.pi
x_cont = np.arange(0, 2*np.pi, 0.1)
amplitude = np.sin(x_cont)

    
x=[2*pi*np.random.random_sample() for i in range(N)]
eta=[np.random.normal(0, 0.1) for i in range(N)]


t = np.sin(x) + eta

#converting tuples into array
x = np.array([x])
t = np.array([t])

x = x.transpose()
t = t.transpose()


def riskFunction(y, t, n):
    return np.sum((y(x)-t)**2)/len(y)


uniform_xs=np.linspace(0,2*np.pi,10000)
integrals=[]

fig = plt.figure(figsize=(18,18))
#level of complexity from 1 to 8
for p in range(1,9,1):
    phi = np.ones((N, 1))
    for i in range(1,p+1,1):
        f = np.power(x,i)
        phi = np.hstack((phi, f))

    inv = np.linalg.pinv(np.matmul(phi.transpose(),phi))
    inv_p = np.matmul(inv, phi.transpose())

    # compute the coefficients
    w_opt = np.matmul(inv_p, t)
    #reversing the array
    w_opt = w_opt[::-1]
    w_opt = w_opt.reshape((p+1,))
    w_opt_list = list(w_opt)
    

    ## polynomial

    poli = np.poly1d(w_opt_list[:])
    
    integral=np.mean((poli(uniform_xs)-np.sin(uniform_xs))**2)
    integrals.append(integral)
    
    ax=fig.add_subplot(3, 3, p)
    #plt.ylim((-1,1))
    plt.title('level of complexity = '+str(p))
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.plot(x_cont, poli(x_cont))
    plt.plot(x_cont, amplitude, c='g')
    plt.scatter(x,t, c='r')
    risk = riskFunction(poli, t, 10)
    risk_array.append(risk)
    
plt.savefig('tex/images/samples-'+str(N)+'.pdf')

fig=plt.figure(figsize=(10,10))
plt.xlabel('level of complexity p')
plt.ylabel('empirical risk')
plt.plot(range(1,9),risk_array)
plt.savefig('tex/images/empirical_risk-'+str(N)+'.pdf')

integrals = [i/(2*np.pi)+0.1 for i in integrals]
fig=plt.figure(figsize=(10,10))
plt.xlabel('level of complexity p')
plt.ylabel('true risk')
plt.plot(range(1,9),integrals)
plt.savefig('tex/images/true_risk-'+str(N)+'.pdf')
