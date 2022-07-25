#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:07:15 2022

@author: selva
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendulum_free(x, t):
    """
    velocity and accelaration of the pendulum
    """
    derX = np.zeros(2)
    derX[0] = x[1]
    l=3 # length of the pendulam
    g=9.8 # accelaration of the gravity in m2/s2
    derX[1] = (-9.81/l)*np.sin(x[0])
    return derX

def pendulum_damped(x, t):
    """
    velocity and accelaration of the pendulum
    """
    derX = np.zeros(2)
    derX[0] = x[1]
    l=3 # length of the pendulam
    g=9.8 # accelaration of the gravity in m2/s2
    damp = 0.3
    derX[1] = (-9.81/l)*np.sin(x[0]) - damp* x[1]
    return derX




"Setting parameters"
t0 = 0 #initial time
tf = 15 #final time
t = np.linspace(t0, tf)

#position or angle of the pendulam at t=0 and #velocity at t=0
xx = np.array([np.pi/3, 0])
#AngTi = np.linspace(-AngRt0, AngRt0)

Y_free = odeint(pendulum_free, xx, t);
Y_damp = odeint(pendulum_damped, xx, t);

plt.figure()
plt.subplot(2,2,1)
plt.plot(t,Y_free[:,0]);
plt.ylabel(r'$y(t)$')
plt.xlabel('time')
plt.subplot(2,2,3)
plt.plot(t,Y_free[:,1]);
plt.ylabel(r'$y(t)$')
plt.xlabel('time')
plt.subplot(2,2,2)
plt.plot(t,Y_damp[:,0]);
plt.ylabel(r'$y(t)$')
plt.xlabel('time')
plt.subplot(2,2,4)
plt.plot(t,Y_damp[:,1]);
plt.ylabel(r'$y(t)$')
plt.xlabel('time')