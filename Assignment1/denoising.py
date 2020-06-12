#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:11:49 2020

"""

import numpy as np
import mnist
import matplotlib.pyplot as plt

def p(x,y,s):
    """Conditional probability of x given y.
    
    The conditional probability of x given y, x is normally distributed with mean y.
    Probability is scaled by sqrt((2*pi)^D*s^2), because that term cancelles
    out for the mean and can be ignored for the argmax, as it is positive.
    y can be a single image or a ndarray of images.
    In the latter case a ndarray of probabilities is returned."""
    if np.ndim(x)==np.ndim(y) and len(x)!=len(y):
        print('ERROR: sizes of vectors do not match')
        return -1
    else:
        prob=np.exp(-1/(2*s**2)*np.sum((y.reshape(-1,np.size(x))-x.flatten())**2,axis=1))
        return prob
    
    
def c_mean(x,Y,s):
    """Conditional mean for given x and Y representing the possible images"""
    return sum(Y*p(x,Y,s)[:,None,None])/sum(p(x,Y,s))


def MAP(x,Y,s):
    """Maximum a priori prediction from Y given x"""
    return Y[np.argmax(p(x,Y,s))]

def make_noisy(x,s):
    """Adds normal random noise with mean 0 and variance s**2 to x"""
    H=len(x)
    W=len(x[0])
    flat=x.flatten()
    img=flat+np.random.normal(0,s**2,H*W)
    img=img.reshape(H,W)
    return img

def arrange_images(ims,rows,cols):
    """Takes a list of rows*cols images and arranges them in a rectangel"""
    for i in range(cols):
        col=ims[10*i]
        for j in range(1,rows):
            col=np.concatenate((col,ims[10*i+j]))
        if i==0:
            images=col
        else:
                images=np.concatenate((images,col),1)
    return images

Y = mnist.load_data("train-images-idx3-ubyte.gz")
X = mnist.load_data("t10k-images-idx3-ubyte.gz")

#normalising the images
X=X/255
Y=Y/255

x_indices=np.random.choice(len(X),100)
y_indices=np.random.choice(len(Y),10000)

X_test=X[x_indices,:,:]
Y_test=Y[y_indices,:,:]

X_noisy=[make_noisy(x,1) for x in X_test]


plt.imsave('tex/images/originals.pdf', arrange_images(X_test,10,10))

for s in [0.25,0.5,1]:
    X_noisy=[make_noisy(x,s) for x in X_test]
    conditional_means=[c_mean(x,Y_test,s) for x in X_noisy]
    maps=[MAP(x,Y_test,s) for x in X_test]
    
    noisy_images=arrange_images(X_noisy,10,10)
    plt.imsave('tex/images/noisy-'+str(int(100*s))+'.pdf',noisy_images)
    
    cm_images=arrange_images(conditional_means,10,10)
    plt.imsave('tex/images/denoised-'+str(int(100*s))+'-mean.pdf',cm_images)
    
    map_images=arrange_images(maps,10,10)
    plt.imsave('tex/images/denoised-'+str(int(100*s))+'-map.pdf',map_images)
