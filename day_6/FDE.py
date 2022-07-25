#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:46:31 2022

@author: selva
"""
import numpy as np
import matplotlib.pyplot as plt



x = np.linspace(-6,6,6000)

#%%

"""
Evaluate the functions
"""
def f_x(x):
    """
    Calculating cos(x) + x * sin(x)
    """
    fx = np.cos(x) + x*np.sin(x)
    return fx

def df_x(x):
    """
    Calculating x * cos(x) 
    """
    dfx = x*np.cos(x)
    return dfx

plt.plot(x,f_x(x), '-b', label=r'$y$')
plt.plot(x, df_x(x), '-g', label=r'$\doty$')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('Functions')
#%%

"""
Evaluate the derivative numerically using finite differences by forward , h=0.25
"""

h = 0.02;

f_x(x)

dfy = np.array([])
xax = np.array([])

xi = -6

while xi < np.max(x):
    dfy = np.append( dfy, (f_x(xi+h)-f_x(xi))/h)
    xax = np.append(xax, xi)
    xi = xi + h
    
plt.plot(xax, dfy, '-r', label = r'$\dotf_{forward}$ & h:'+str(h))

#%%

"""
Evaluate the derivative numerically using finite differences by forward , h=0.25
"""

hb = h;

f_x(x)

dfy_b = np.array([])
xax_b = np.array([])

xi = 6

while xi > np.min(x):
    dfy_b = np.append( dfy_b, (f_x(xi)-f_x(xi-hb))/hb)
    xax_b = np.append(xax_b, xi)
    xi = xi - hb
    
plt.plot(xax_b, dfy_b, '-k', label = r'$\dotf_{backward}$ & h:'+str(hb))
plt.legend()

#%%

"""
Evaluate the derivative numerically using finite differences by central value , h=0.25
"""  

hc =h;

f_x(x)

dfy_c = np.array([])
xax_c = np.array([])

xi = 6

while xi > np.min(x):
    dfy_c = np.append( dfy_c, (f_x(xi+hc)-f_x(xi-hc))/(2*hc))
    xax_c = np.append(xax_c, xi)
    xi = xi - hc
    
plt.plot(xax_c, dfy_c, '-y', label = r'$\dotf_{central}$ & h:'+str(hc))
plt.legend()












