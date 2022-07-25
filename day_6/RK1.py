#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:12:40 2022

@author: selva
"""

"""
Euler Method / Ranga-kutta 1st order for ordinary differential equation
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def RHS(y,t):
    """
    ODE right hand side
    """
    return -2*y

"Set the problem"
y0=3 #initial condition
t0=0 #initial time
tf=2 #final time

"Evaluate exact solution"
time = np.linspace(t0, tf)
y_true = odeint(RHS, y0, time)

fig1 = plt.figure()
plt.plot(time, y_true, 'ok-', linewidth=2, label='Truth')
plt.grid()
plt.ylabel(r'$y(t)$')
plt.xlabel('time')


"Numerical Integration for 1st Order"
yt0 = 3
h = 0.2
ti=np.min(time)
xax = np.array([])
yti = np.array([yt0])
while ti<np.max(time):
    yt1 = yt0 + ( h*RHS(yt0, 0))
    yti = np.append(yti, yt1)
    yt0 = yt1
    xax = np.append(xax, ti)
    ti = ti+h
    
plt.plot(xax, yti[:-1], 'sr-', linewidth=2, label='Ranga-Kutta 1')


"Numerical Integration for 2nd Order"
cV=3
cT = np.min(time)
current_value = 3
ti=np.min(time)
xax2 = np.array([ti])
yti2 = np.array([cV])
while ti<np.max(time):

    k1 = RHS(cV, cT)
    k2 = RHS(cV + k1*(h/2), cT+(h/2))
    k3 = RHS(cV + k2*(h/2), cT+(h/2))
    k4 = RHS(cV + k3*h, cT+h)
    nextV = cV + (k1+2*k2+2*k3+k4)*(h/6)
    yti2 = np.append(yti2, nextV)
    cV = nextV
    ti = ti+h
    xax2 = np.append(xax2, ti)

    # k1 = RHS(current_value, ti)
    # k2 = RHS(current_value+(h/2)*k1, ti+(h/2))
    # nextV = current_value+k2*h

    # yti2 = np.append(yti2, nextV)
    # current_value = nextV
    
    # xax2 = np.append(xax2, ti)
    # ti = ti+h
    
    
plt.plot(xax2[:-1], yti2[:-1], '+b-', linewidth=2, label='Ranga-Kutta 4')
plt.legend()
sys.exit()





