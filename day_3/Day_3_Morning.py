#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welcome to Space Weather Simulation Summer School Day 3

Today, we will be working with various file types, doing some simple data 
manipulation and data visualization

We will be using a lot of things that have been covered over the last two days 
with minor vairation.

Goal: Getting comfortable with reading and writing data, doing simple data 
manipulation, and visualizing data.

Task: Fill in the cells with the correct codes

@author: Peng Mun Siew
"""

#%% 
"""
This is a code cell that we can use to partition our data. (Similar to Matlab cell)
We hace the options to run the codes cell by cell by using the "Run current cell" button on the top.
"""
print ("Hello World")

#%%
"""
Writing and reading numpy file
"""
# Importing the required packages
import numpy as np

# Generate a random array of dimension 10 by 5
data_arr = np.random.randn(10,5)
print(data_arr)

# Save the data_arr variable into a .npy file
np.save('test_np_save.npy', data_arr)

#Load data from a .npy file
data_arr_loaded = np.load('test_np_save.npy')

#Verification that the loaded data matches the initial data exactly
print(np.equal(data_arr, data_arr_loaded))
print(data_arr==data_arr_loaded)

#%%
"""
Writing and reading numpy zip archive/file
"""
# Generate a second random array of dimension 8 by 1
data_arr2 = np.random.randn(8,1)
print(data_arr2)

# Save the data_arr and data_arr2 variables into a .npz file
np.savez('test_savez.npz', data_arr, data_arr2)

#load the npz file
npzfile=np.load('test_savez.npz')

#loaded file is not a np file to print.
print(npzfile)

#To inspect the name of the variable
print('Variale names within this file:', sorted(npzfile.files))

# We will then be able to use the variable name as a key to access the data
print(npzfile['arr_0'])

#Verification that the loaded data matches the initial data exactly
print((data_arr==npzfile['arr_0']).all())
print((data_arr2==npzfile['arr_1']).all())

#%%
"""
Error and exception
"""



#%%
"""
Loading data from Matlab
"""

# Import required packages
import numpy as np
from scipy.io import loadmat

dir_density_Jb2008 = '/Volumes/Data/Coding/SpaceWeather/CU2022/JB2008/2002_JB2008_density.mat'

# Load Density Data
try:
    loaded_data = loadmat(dir_density_Jb2008)
    print (loaded_data)
except:
    print("File not found. Please check your directory")

# Uses key to extract our data of interest
JB2008_dens = loaded_data['densityData']

# The shape command now works
print(JB2008_dens.shape)

#%%
"""
Data visualization I

Let's visualize the density field for 400 KM at different time.
"""
# Import required packages
import matplotlib.pyplot as plt

# Before we can visualize our density data, we first need to generate the discretization grid of the density data in 3D space. We will be using np.linspace to create evenly sapce data between the limits.

localSolarTimes_JB2008 = np.linspace(0,24,24)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0]
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
nofLat_JB2008 = latitudes_JB2008.shape[0]

# We can also impose additional constratints such as forcing the values to be integers.
time_array_JB2008 = np.linspace(0,8759,20, dtype = int)

# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008,8760), order='F') # Fortran-like index order

#Look for data that correspond to an altitude of 400 km
alt = 400;
hi = np.where(altitudes_JB2008==alt)

# Create a canves to plot our data on.
fig, axs = plt.subplots(20, figsize=(9, 18), sharex=True)

for ik in range(20):
    cs = axs[ik].contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_array_JB2008[ik]].squeeze().T)
    axs[ik].set_title('JB2008 density at 400 km, t = {} hours'.format(time_array_JB2008[ik]), fontsize=12)
    axs[ik].set_ylabel("Latitudes", fontsize=18)
    axs[ik].tick_params(axis='both', which='major', labelsize=16)
    
    #Make a colorbar for the Contour call
    cbar = fig.colorbar(cs, ax=axs[ik])
    cbar.ax.set_ylabel('Density')

axs[ik].set_xlabel('Local Solar Time', fontsize=18)


#%%
"""
Assignment 1

Can you plot the mean density for each altitude at February 1st, 2002?
"""

# First identidy the time index that corresponds to  February 1st, 2002. Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = JB2008_dens_reshaped[:,:,:,time_index]

print(dens_data_feb1.shape)

DenPrf = np.mean(np.mean(dens_data_feb1, axis=1), axis=0)
#DenPrfx = np.mean(dens_data_feb1, axis=(0,1))
DenPrfP = np.zeros([len(altitudes_JB2008)]);
for ri in range(len(altitudes_JB2008)):
    DenPrfP[ri] = np.mean(dens_data_feb1[:,:,ri])

eta = 1e-20;
print((DenPrfP-DenPrf<eta).all())
print((DenPrfP-np.asarray(DenPrf)<eta).all())


fig, axs = plt.subplots(1, figsize=(9, 18), sharex=True)
axs.plot(DenPrf, altitudes_JB2008, 'ob', label = 'Average from mean of mean')
axs.plot(DenPrfP, altitudes_JB2008, '-+r', label = 'Average from for-loop')
axs.set_xscale('log'); axs.grid(True);
axs.set_title('JB2008 density Vs Altitue, t = {} hours'.format(time_index), fontsize=12)
axs.set_xlabel("Density", fontsize=18)
axs.set_ylabel("Altitude (km)", fontsize=18)
plt.legend()
#%%
"""
Data Visualization II

Now, let's us work with density data from TIE-GCM instead, and plot the density field at 310km
"""
# Import required packages
import h5py
Ldat = h5py.File('/Volumes/Data/Coding/SpaceWeather/CU2022/TIEGCM/2002_TIEGCM_density.mat');
#This is a hdf5 data object, some similarity with a dictionary
print('Key within dataset:', list(Ldat.keys()))

Dens_GCM = (10**np.array(Ldat['density'])*1000).T; #convert from g/cm3 to kg/m3
Alt_GCM = np.array(Ldat['altitudes']).flatten();
Lat_GCM = np.array(Ldat['latitudes']).flatten();
Solar_tme_GCM = np.array(Ldat['localSolarTimes']).flatten();

NoAlt = Alt_GCM.shape[0]
NoLat = Lat_GCM.shape[0]
NoLst = Solar_tme_GCM.shape[0]

# We can also impose additional constratints such as forcing the values to be integers.
time_GCM = np.linspace(0,8759,20, dtype = int)

# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
Dens_GCM_M = np.reshape(Dens_GCM,(NoLst,NoLat,NoAlt,8760), order='F') # Fortran-like index order

#Look for data that correspond to an altitude of 400 km
alt = 310;
hi = np.where(Alt_GCM==alt)

# Create a canves to plot our data on.
fig, axs = plt.subplots(2, figsize=(9, 18), sharex=True)

for ik in range(2):
    cs = axs[ik].contourf(Solar_tme_GCM, Lat_GCM, Dens_GCM_M[:,:,hi,time_GCM[ik]].squeeze().T)
    axs[ik].set_title('TIE-GCM density at '+str(alt)+' km, t = {} hours'.format(time_GCM[ik]), fontsize=12)
    axs[ik].set_ylabel("Latitudes", fontsize=18)
    axs[ik].tick_params(axis='both', which='major', labelsize=16)
    
    #Make a colorbar for the Contour call
    cbar = fig.colorbar(cs, ax=axs[ik])
    cbar.ax.set_ylabel('Density')

axs[ik].set_xlabel('Local Solar Time', fontsize=18)

#%%
"""
Can you plot the mean density for each altitude at February 1st, 2002?
"""

# First identidy the time index that corresponds to  February 1st, 2002. Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1_GCM  = Dens_GCM_M[:,:,:,time_index]
print(Dens_GCM_M.shape)

DenPrfg = np.mean(np.mean(dens_data_feb1_GCM , axis=1), axis=0)
#DenPrfx = np.mean(dens_data_feb1, axis=(0,1))
DenPrfP_GCM = np.zeros([len(Alt_GCM)]);
for ri in range(len(Alt_GCM)):
    DenPrfP_GCM[ri] = np.mean(dens_data_feb1_GCM [:,:,ri])

eta = 1e-20;
print((DenPrfP_GCM-DenPrfg<eta).all())
print((DenPrfP_GCM-np.asarray(DenPrfg)<eta).all())


fig, axs = plt.subplots(1, figsize=(9, 18), sharex=True)
axs.plot(DenPrfP_GCM, Alt_GCM, 'ob', label = 'TIE-GCM')
axs.plot(DenPrfP, altitudes_JB2008, '-+r', label = 'JB2008')
axs.set_xscale('log'); axs.grid(True);
axs.set_title('TIE-GCM density Vs Altitue, t = {} hours'.format(time_index), fontsize=12)
axs.set_xlabel("Density", fontsize=18)
axs.set_ylabel("Altitude (km)", fontsize=18)
plt.legend()
#%%

#%%
"""
Data Interpolation (1D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy import interpolate

# Let's first create some data for interpolation
x = np.arange(0, 10)
y = np.exp(-x/3.0)
#Generate 1D interpolation function
interp_1D = interpolate.interp1d(x, y)
interp_cubic = interpolate.interp1d(x, y, kind='cubic')

xN = np.linspace(0, 9, 10)

yN1D = interp_1D(xN)
yN1D_c = interp_cubic(xN)

plt.figure(),
plt.plot(x,y, '-ob', label='initial')
plt.plot(xN,yN1D, '-+r', label='interpolated')
plt.plot(xN,yN1D_c, '-+r', label='interpolated-cubic')
plt.title('Interpolation in 1D', fontsize=12)
plt.legend()
#%%

"""
Data Interpolation (3D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator

# First create a set of sample data that we will be using 3D interpolation on
def function_1(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

sample_data = function_1(xg, yg, zg)

#Generate Interpolant (interpolating function)
interp_fun1 = RegularGridInterpolator((x,y,z), sample_data)

pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
print('Using interpolation method:', interp_fun1(pts))
print('From true function:', function_1(2.1, 6.2,8.3))

#Generate interpolatant for GCM data
time_GCMa = np.linspace(0,8760,8760, dtype = int)
interp_fun1gcm = RegularGridInterpolator((Solar_tme_GCM,Lat_GCM,Alt_GCM), dens_data_feb1_GCM, bounds_error=False, fill_value=None)
interp_fun1_JB = RegularGridInterpolator((localSolarTimes_JB2008,latitudes_JB2008,altitudes_JB2008), dens_data_feb1, bounds_error=False, fill_value=None)

def intp3d_gcm(t,l,a):
    return interp_fun1gcm((t,l,a))
Den_3D_GCM = np.vectorize(intp3d_gcm)

def intp3d_jb(t,l,a):
    return interp_fun1_JB((t,l,a))
Den_3D_JB = np.vectorize(intp3d_jb)

lat3d = np.linspace(-90, 90, 180);
tme3d = np.linspace(0, 24, 180);
#alt3d = np.linspace(100, 800, 180);
Lat_Tme = np.meshgrid(lat3d, tme3d);

###-----
lat3dA = np.linspace(-90, 90, 1800);
tme3dA = np.linspace(0, 24, 1000);

VdenJB = np.zeros((len(tme3dA),len(lat3dA)))
for ti in range(len(tme3dA)):
    for li in range(len(lat3dA)):
        VdenJB[ti,li] = interp_fun1_JB((tme3dA[ti],lat3dA[li],400))

VdenGCM = np.zeros((len(tme3dA),len(lat3dA)))
for ti in range(len(tme3dA)):
    for li in range(len(lat3dA)):
        VdenGCM[ti,li] = interp_fun1gcm((tme3dA[ti],lat3dA[li],400))

#VdenGCM = Den_3D_GCM(lat3d, tme3d, alt3d)
#VdenJB = Den_3D_JB(lat3d, tme3d, alt3d)
#VdenJB_M = Den_3D_JB(Lat_Tme[0], Lat_Tme[1], 400);
#plt.figure(), plt.contourf(Lat_Tme[1], Lat_Tme[0], VdenJB_M)

# Create a canves to plot our data on.
fig, axs = plt.subplots(2, figsize=(9, 18), sharex=True)
ik=0
cs = axs[ik].contourf(tme3dA, lat3dA, VdenJB.T)
axs[ik].set_title('JB2008 density at 400 km, t = {} hours'.format(time_array_JB2008[ik]), fontsize=12)
axs[ik].set_ylabel("Latitudes", fontsize=18)
axs[ik].tick_params(axis='both', which='major', labelsize=16)
#Make a colorbar for the Contour call
cbar = fig.colorbar(cs, ax=axs[ik])
cbar.ax.set_ylabel('Density')
axs[ik].set_xlabel('Local Solar Time', fontsize=18)
ik=1
cs = axs[ik].contourf(tme3dA, lat3dA, VdenGCM.T)
axs[ik].set_title('TIE-GCM density at 400 km, t = {} hours'.format(time_GCM[ik]), fontsize=12)
axs[ik].set_ylabel("Latitudes", fontsize=18)
axs[ik].tick_params(axis='both', which='major', labelsize=16)
#Make a colorbar for the Contour call
cbar = fig.colorbar(cs, ax=axs[ik])
cbar.ax.set_ylabel('Density')
axs[ik].set_xlabel('Local Solar Time', fontsize=18)


print('TIE-GCM density at (lst=20hrs, lat=12 deg and alt = 400 km)', interp_fun1gcm((20,12,400)) )
#%%
"""
Saving mat file

Now, let's us look at how to we can save our data into a mat file
"""
# Import required packages
from scipy.io import savemat

a = np.arange(20)
mdic = {"a": a, "label": "experiment", "v":1} # Using dictionary to store multiple variables
savemat("matlab_matrix.mat", mdic)

#%%
"""
Assignment 2 (a)

The two data that we have been working on today have different discretization grid.
Use 3D interpolation to evaluate the TIE-GCM density field at 400 KM on February 1st, 2002, with the discretized grid used for the JB2008 ((nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008).
"""





#%%
"""
Assignment 2 (b)

Now, let's find the difference between both density models and plot out this difference in a contour plot.
"""





#%%
"""
Assignment 2 (c)

In the scientific field, it is sometime more useful to plot the differences in terms of mean absolute percentage difference/error (MAPE). Let's plot the MAPE for this scenario.
"""



