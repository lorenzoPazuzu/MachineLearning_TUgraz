#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:14:20 2020

@author: daniel
"""

import numpy as np
import matplotlib.pyplot as plt

def prox_sub_grad(w,x,alpha,l):
    _phi=phi(x)
    gamma=np.ones((len(x[0])))
    gamma[0]=1e-6
    for i in range(10000):
        t=np.reshape(x[:,2],(len(x),1))
        g=-np.repeat(t,3,axis=1)*_phi
        t=np.reshape(t,(len(x)))
        g[t*np.matmul(_phi,w)>=1]=0
        g=np.mean(g,axis=0)
        w=w-alpha*g
        w=w/(1+alpha*l*gamma)
    return w

def phi(x):
    _phi=np.concatenate((np.ones((len(x),1)),x[:,0:2]),axis=1)
    return _phi

mu1=[6.5,2]
sigma1=[[0.8,0],[0,0.7]]

mu2=[1.5,2]
sigma2=[[1,0],[0,0.2]]

simple1=np.random.multivariate_normal(mu1,sigma1,100)
simple1=np.concatenate((simple1,np.ones((100,1))),axis=1)

simple2=np.random.multivariate_normal(mu2,sigma2,100)
simple2=np.concatenate((simple2,-np.ones((100,1))),axis=1)

simple=np.concatenate((simple1,simple2))

plt.scatter(simple1[:,0],simple1[:,1])
plt.scatter(simple2[:,0],simple2[:,1],color='red')
plt.savefig('tex/images/simple.pdf')
plt.show()

sd=0.01

eta1=np.random.normal(0,sd,200)
eta2=np.random.normal(0,sd,200)

theta=np.random.uniform(0,np.pi,200)

x1=np.reshape(np.cos(theta[:100])+eta1[:100],(100,1))
y1=np.reshape(np.sin(theta[:100])+eta2[:100],(100,1))

x2=np.reshape(1-np.cos(theta[100:])+eta1[100:],(100,1))
y2=np.reshape(1-np.sin(theta[100:])+eta2[100:],(100,1))

moon1=np.concatenate((x1,y1,np.ones((100,1))),axis=1)
moon2=np.concatenate((x2,y2,-np.ones((100,1))),axis=1)

moon=np.concatenate((moon1,moon2))

plt.scatter(moon1[:,0],moon1[:,1])
plt.scatter(moon2[:,0],moon2[:,1],color='red')
plt.savefig('tex/images/moon.pdf')
plt.show()

w=np.ones((3))
w=prox_sub_grad(w,simple,0.01,0.5)
w/=w[1]
ys=[np.min(simple[:,1]),np.max(simple[:,1])]
xs=-w[0]-np.multiply(w[2],ys)

left=simple[np.matmul(phi(simple),w)<=0]
right=simple[np.matmul(phi(simple),w)>=0]

closest1=np.argmin(np.matmul(phi(simple1),w))
closest2=np.argmax(np.matmul(phi(simple2),w))

left_offset=simple1[closest1,0]-(-w[0]-np.multiply(w[2],simple1[closest1,1]))
right_offset=simple2[closest2,0]-(-w[0]-np.multiply(w[2],simple2[closest2,1]))

plt.scatter(simple1[:,0],simple1[:,1])
plt.scatter(simple2[:,0],simple2[:,1],color='red')
plt.scatter(simple1[closest1,0], simple1[closest1,1],facecolors='none', edgecolors='black',s=150)
plt.scatter(simple2[closest2,0], simple2[closest2,1],facecolors='none', edgecolors='black',s=150)
plt.plot(xs,ys,c='black',linewidth=2)
plt.plot(xs+left_offset,ys,c='black',linewidth=1)
plt.plot(xs+right_offset,ys,c='black',linewidth=1)
plt.savefig('tex/images/simple-line.pdf')
plt.show()

###############################################################################

w=np.ones((3))
w=prox_sub_grad(w,moon,0.01,1)
w/=w[1]
ys=[np.min(moon[:,1]),np.max(moon[:,1])]
xs=-w[0]-np.multiply(w[2],ys)

left=moon[np.matmul(phi(moon),w)<=0]
right=moon[np.matmul(phi(moon),w)>=0]

closest1=np.argmax(np.matmul(phi(moon1),w))
closest2=np.argmin(np.matmul(phi(moon2),w))

left_offset=moon1[closest1,0]-(-w[0]-np.multiply(w[2],moon1[closest1,1]))
right_offset=moon2[closest2,0]-(-w[0]-np.multiply(w[2],moon2[closest2,1]))

plt.scatter(moon1[:,0],moon1[:,1],marker='x')
plt.scatter(moon2[:,0],moon2[:,1],color='red',marker='x')
plt.scatter(moon1[closest1,0], moon1[closest1,1],facecolors='none', edgecolors='black',s=150)
plt.scatter(moon2[closest2,0], moon2[closest2,1],facecolors='none', edgecolors='black',s=150)
plt.plot(xs,ys,c='black',linewidth=2)
plt.plot(xs+left_offset,ys,c='black',linewidth=1)
plt.plot(xs+right_offset,ys,c='black',linewidth=1)
plt.savefig('tex/images/moon-line.pdf')
plt.show()