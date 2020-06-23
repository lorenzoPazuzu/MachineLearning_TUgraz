#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 18:52:02 2020

@author: daniel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:14:20 2020
@author: daniel, lorenzo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

sigma=0.5
epsilon=0.1
np.random.seed(10)


def prox_sub_grad(w,x,alpha,l,eps=1e-2):
    _phi=phi(x)
    gamma=np.ones((len(x[0])))
    gamma[0]=1e-6
    w_old=w+1
    i=0
    while np.linalg.norm(w-w_old)>eps or i <10000:
        t=np.reshape(x[:,2],(len(x),1))
        g=-np.repeat(t,3,axis=1)*_phi
        t=np.reshape(t,(len(x)))
        g[t*np.matmul(_phi,w)>=1]=0
        g=np.mean(g,axis=0)
        w_old=w
        w=w-alpha*g
        w=w/(1+alpha*l*gamma)
        i+=1
    return w

def phi(x):
    _phi=np.concatenate((np.ones((len(x),1)),x[:,0:2]),axis=1)
    return _phi

def kernel_G(x, sigma):
    n=len(x)
    x=np.repeat(x[:,:,np.newaxis],n,axis=2)
    x2=np.swapaxes(x,0,2)
    k=x-x2
    k=k**2
    k=np.sum(k,axis=1)
    k=k/-(2*sigma)
    k=np.exp(k)
    return k

def kernel_G2(x1,x2,sigma):
    k=x1-x2
    k=k**2
    k=np.sum(k,axis=-1)
    k=k/-(2*sigma)
    k=np.exp(k)
    return k

def gradient(a, x, t):
    x2=np.repeat(x[:,np.newaxis,:],len(x),axis=1)
    x2=np.swapaxes(x2,0,1)
    x=np.repeat(x[:,np.newaxis,:],len(x),axis=1)
    k=kernel_G2(x,x2,sigma)
    g=1-t * (k @ (a*t))
    return g

def FISTA(x, t,eps=1e-2):
    j1=1
    a1=a0=np.zeros((len(x)))
    a_old=a1+1
    alpha=0.001
    D_array=[]
    i=0
    while np.abs(D(a1,t,phi(x),sigma)-D(a_old,t,phi(x),sigma))>eps or i<1000:
        j0=j1
        j1=(1+np.sqrt(1+4*(j0**2)))/2
        a_tilde=a1+((j0-1)/j1)*(a1-a0)
        a_tilde=np.reshape(a_tilde,(len(x)))
        a0=a1
        compare = a_tilde+alpha*gradient(a_tilde,x,t)
        #print(gradient(a_tilde,x,t)/approx_fprime(a_tilde, D, 1e-2, t,x,sigma))
        #compare = a_tilde+alpha*approx_fprime(a_tilde, D, 1e-6, t,phi(x),sigma)
        a_old=a1
        a1=np.max(np.concatenate((np.zeros((len(x),1)),compare[:,np.newaxis]),axis=1),axis=1)

        D_array.append(D(a1,t,phi(x),sigma))
        i+=1
        #print(np.abs(D(a1,t,phi(x),sigma)-D(a_old,t,phi(x),sigma)))
    return a1,D_array

def y(x,X,a,t,sigma):
    x=np.transpose(x)
    x=np.repeat(x[:,:,np.newaxis,:],len(X),axis=2)
    X=np.repeat(X[np.newaxis,:,:],len(x),axis=0)
    X=np.repeat(X[:,np.newaxis,:,:],len(x[0]),axis=1)
    k=kernel_G2(x,X,sigma)
    _y=np.multiply(a,t)
    _y=np.multiply(_y,k)
    _y=np.sum(_y,axis=-1)
    return np.transpose(_y)

def D(a, t, phi, sigma):
    part = np.sum(a)
    t = np.asmatrix(t)
    a = np.asmatrix(a)
    part2 = np.dot(t.T, t)
    part3 = np.dot(a.T, t)
    part2 = part2.reshape(len(phi)**2, 1)
    part3 = part3.reshape(len(phi)**2, 1)
    part4 = np.multiply(part2, part3)
    k = kernel_G(phi, sigma).reshape(len(phi)**2, 1)
    part4 = np.multiply(part4, k)
    part4 = np.sum(part4)
    return -0.5*part4+part

def f(w,x,y):
    return w[0]+w[1]*x+w[2]*y


mu1=[6.5,2]
sigma1=[[0.8,0],[0,0.7]]

mu2=[1.5,2]
sigma2=[[1,0],[0,0.2]]

simple1=np.random.multivariate_normal(mu1,sigma1,100)
simple1=np.concatenate((simple1,np.ones((100,1))),axis=1)

simple2=np.random.multivariate_normal(mu2,sigma2,100)
simple2=np.concatenate((simple2,-np.ones((100,1))),axis=1)

simple=np.concatenate((simple1,simple2))

plt.scatter(simple1[:,0],simple1[:,1],marker='x')
plt.scatter(simple2[:,0],simple2[:,1],color='red',marker='x')
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

plt.scatter(moon1[:,0],moon1[:,1],marker='x')
plt.scatter(moon2[:,0],moon2[:,1],color='red',marker='x')
plt.savefig('tex/images/moon.pdf')
plt.show()

w=np.ones((3))
w=prox_sub_grad(w,simple,3,0.0001)

x=np.linspace(np.min(simple[:,0]),np.max(simple[:,0]),50)
_y=np.linspace(np.min(simple[:,1]),np.max(simple[:,1]),50)

xx,yy=np.meshgrid(x,_y)
z=f(w,xx,yy)

support_vectors=simple[np.abs(f(w,simple[:,0],simple[:,1])-1)<epsilon]
support_vectors=np.concatenate((support_vectors,simple[np.abs(f(w,simple[:,0],simple[:,1])+1)<epsilon]))

plt.scatter(simple1[:,0],simple1[:,1],marker='x')
plt.scatter(simple2[:,0],simple2[:,1],color='red',marker='x')
plt.scatter(support_vectors[:,0], support_vectors[:,1],facecolors='none', edgecolors='black',s=150)
plt.contour(xx,yy,z,levels=[-1,0,1])

plt.savefig('tex/images/simple-line.pdf')
plt.show()

###############################################################################

w=np.ones((3))
w=prox_sub_grad(w,moon,3,0.0001,1)

x=np.linspace(np.min(moon[:,0]),np.max(moon[:,0]),50)
_y=np.linspace(np.min(moon[:,1]),np.max(moon[:,1]),50)

xx,yy=np.meshgrid(x,_y)
z=f(w,xx,yy)

support_vectors=moon[np.abs(f(w,moon[:,0],moon[:,1])-1)<epsilon]
support_vectors=np.concatenate((support_vectors,moon[np.abs(f(w,moon[:,0],moon[:,1])+1)<epsilon]))

plt.scatter(moon1[:,0],moon1[:,1],marker='x')
plt.scatter(moon2[:,0],moon2[:,1],color='red',marker='x')
plt.scatter(support_vectors[:,0], support_vectors[:,1],facecolors='none', edgecolors='black',s=150)
plt.contour(xx,yy,z,levels=[-1,0,1])
plt.savefig('tex/images/moon-line.pdf')
plt.show()

###############################################################################

a,D_array=FISTA(moon[:,:2],moon[:,2],100)


x=np.linspace(-1,2,100)
_y=np.linspace(0,1,100)
xy=np.meshgrid(x,_y)
X=moon[:,:2]
t=moon[:,2]
z=y(xy,X,a,t,sigma)

X2=np.swapaxes([X],0,2)
X2=np.swapaxes(X2,1,2)

support_vectors=moon[np.abs(np.reshape(y(X2,X,a,t,sigma),(200))-1)<epsilon]
support_vectors=np.concatenate((support_vectors,moon[np.abs(np.reshape(y(X2,X,a,t,sigma),(200))+1)<epsilon]))

plt.scatter(moon1[:,0],moon1[:,1],marker='x')
plt.scatter(moon2[:,0],moon2[:,1],color='red',marker='x')
plt.scatter(support_vectors[:,0], support_vectors[:,1],facecolors='none', edgecolors='black',s=150)
plt.contour(xy[0],xy[1],z,levels=[-1,0,1])
plt.title('Decision boundary for sigma = %f' % sigma)
plt.savefig('tex/images/decision_boundary_2.pdf')
plt.show()

### D on iterations
plt.plot(D_array)
plt.title('D for Simple dataset, sigma = %f' % sigma)
plt.xlabel('Iterations')
plt.savefig('tex/images/Simple_D_iterations.pdf')
plt.show()
