#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:30:14 2020

@author: daniel
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

pi=np.pi
for s in [1,5,10]:
    for a in [0,0.5,0.9]:
        title='sigma^2='+str(s**2)+', alpha='+str(a)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        # Make data.
        x = np.linspace(-3*s,3*s,100)
        y = np.linspace(-3*s,3*s,100)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = 1/(2*pi*s**2*np.sqrt(1-a**2))*np.exp(-(X**2+Y**2-2*X*Y*a)/(2*s**2*(1-a**2)))
        
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # Customize the z axis.
        #ax.set_zlim(-2*s, 2*s)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))#show 4 digits on z-axis
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(title)
        plt.savefig('tex/images/plot-'+str(s)+'-'+str(int(10*a))+'-side.pdf')
        
        ax.view_init(azim=0, elev=90)
        ax.zaxis.set_ticklabels([])
        plt.gcf()
        plt.savefig('tex/images/plot-'+str(s)+'-'+str(int(10*a))+'-top.pdf')
        
        
    px=1/np.sqrt(2*pi*s**2)*np.exp(-x**2/(2*s**2))
    fig=plt.figure()
    plt.plot(x,px)
    plt.title('sigma^2='+str(s**2))
    plt.savefig('tex/images/plot-x-'+str(s)+'.pdf')