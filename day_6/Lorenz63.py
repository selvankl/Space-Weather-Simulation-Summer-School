#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
<<<<<<< HEAD
Created on Thu Jul 21 11:53:19 2022

@author: Simone Servadio
"""
"""This file solves the Lorenz 63 system"""
"""simplified mathematical model for atmospheric convection 
The equations relate the properties of a two-dimensional fluid layer uniformly 
 warmed from below and cooled from above. In particular, the equations describe 
 the rate of change of three quantities with respect to time: x is proportional 
 to the rate of convection, y to the horizontal temperature variation, and z to 
 the vertical temperature variation. The constants σ, ρ, and β are system parameters 
 proportional to the Prandtl number, Rayleigh number, and certain physical 
 dimensions of the layer itself."""


#%matplotlib qt
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 




def Lorenz63(x,t,sigma,rho,beta):
    """The Lorenz63 system for different coefficients"""
    xdot = np.zeros(3)
    xdot[0] = sigma*(x[1]-x[0]);
    xdot[1] = x[0]*(rho-x[2])-x[1];
    xdot[2] = x[0]*x[1] - beta*x[2];
    return xdot


"Set parameters"
rho = 28
sigma = 10
beta = 8/3

x0 = 5*np.ones(3); #initial state
t_in = 0 #initial time
t_fin = 20 #final time
n_points = 1000
time = np.linspace(t_in,t_fin,n_points) #time vector

# Solve ODE
y = odeint(Lorenz63,x0,time,args = (sigma,rho,beta)) #propagation

# Display
fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(y[:,0], y[:,1], y[:,2],'b') #3d plottingsolution
#sys.exit()




=======
Created on Mon Jul 25 16:18:04 2022

@author: selva
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def Lorenz63(xx, t, sig, rho, beta ):
    """
    Lorenz63 system for atmospheric convection
    """
    derX = np.zeros(3)
    derX[0] = sig*(xx[1]-xx[0])
    derX[1] = xx[0] * (rho-xx[2]) - xx[1]
    derX[2] = xx[0]*xx[1] - beta*xx[2]

    return derX

sig = 10
rho = 28
beta = 8/3
t0 = 0 #initial time
tf = 20 #final time
t = np.linspace(t0, tf,  1000)
x1 = np.linspace(-20, 20, 20)
y1 = np.linspace(-30, 30, 20)
z1 = np.linspace(0, 50, 20)
#xx = np.array([5,5,5])
#Y_free = odeint(Lorenz63, xx, t, args=(sig, rho, beta ));

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx = np.array([x1,y1,z1])
Y_free = np.zeros((len(t), 3, len(z1)))
for ri in range(len(z1)):
    Y_free[:, :, ri] = odeint(Lorenz63, xx[:,ri], t, args=(sig, rho, beta ));
    ax.plot3D(Y_free[:,0, ri], Y_free[:,1, ri], Y_free[:,2, ri])
>>>>>>> 31a4548 (merge)





"Random Initial Condition"
n_ic = 20 #number of initial conditions
x0s = np.zeros((3,n_ic))

#Initial condition domain
x0s[0,:] = 20 * 2 * (np.random.rand(n_ic) - 0.5);
x0s[1,:] = 30 * 2 * (np.random.rand(n_ic) - 0.5);
x0s[2,:] = 50 * np.random.rand(n_ic);


fig2 = plt.figure()
ax = plt.axes(projection='3d')
for i in range(n_ic):
    y = odeint(Lorenz63,x0s[:,i],time,args = (sigma,rho,beta)) #propagation
    ax.plot3D(y[:,0], y[:,1], y[:,2]) #visualization
    
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
sys.exit()













<<<<<<< HEAD





=======
Created on Mon Jul 25 16:18:04 2022

@author: selva
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def Lorenz63(xx, t, sig, rho, beta ):
    """
    Lorenz63 system for atmospheric convection
    """
    derX = np.zeros(3)
    derX[0] = sig*(xx[1]-xx[0])
    derX[1] = xx[0] * (rho-xx[2]) - xx[1]
    derX[2] = xx[0]*xx[1] - beta*xx[2]

    return derX

sig = 10
rho = 28
beta = 8/3
t0 = 0 #initial time
tf = 20 #final time
t = np.linspace(t0, tf,  1000)
x1 = np.linspace(-20, 20, 20)
y1 = np.linspace(-30, 30, 20)
z1 = np.linspace(0, 50, 20)
#xx = np.array([5,5,5])
#Y_free = odeint(Lorenz63, xx, t, args=(sig, rho, beta ));

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx = np.array([x1,y1,z1])
Y_free = np.zeros((len(t), 3, len(z1)))
for ri in range(len(z1)):
    Y_free[:, :, ri] = odeint(Lorenz63, xx[:,ri], t, args=(sig, rho, beta ));
    ax.plot3D(Y_free[:,0, ri], Y_free[:,1, ri], Y_free[:,2, ri])
>>>>>>> e23347c (Lorenz63 model)
=======
>>>>>>> 31a4548 (merge)
