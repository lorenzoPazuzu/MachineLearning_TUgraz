#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:29:40 2020

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


x = np.linspace(-10,10,500)
y = np.linspace(-10,10,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y


def myMulti(si, alpha):
    mean = [0,0]
    cov = [[si,alpha*(si)], [alpha*(si), si]]
    return multivariate_normal(mean,cov)


s_array = [1, 5, 10]
a_array = [0, 0.5, 0.95]


#Make a 3D plot
fig = plt.figure(figsize=(20, 90))

i=1
for s in s_array:
    for a in a_array:
        ax = fig.add_subplot(9, 2, i, projection='3d')
        ax.title.set_text('var = '+str(s)+' , alpha = '+str(a))
        surf = ax.plot_surface(X, Y, myMulti(s,a).pdf(pos),cmap='viridis',linewidth=0)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        ax = fig.add_subplot(9, 2, i+1)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.contour(X, Y, myMulti(s,a).pdf(pos), 50)
        
        i=i+2

plt.savefig('test.pdf')
