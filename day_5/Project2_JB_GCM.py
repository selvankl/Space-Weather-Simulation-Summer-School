#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:56:33 2022

@author: selva
"""

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
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import locale
locale.getpreferredencoding(False)

def Omni_data_read(rdir, index):
    "This code reads the omni data"
    data_dic = {"time":[], 
                "year":[], 
                "DayN":[], 
                "Hour":[], 
                "Minute":[], 
                "data":[]}
    with open(rdir, 'r') as f:
        for lindat in f:
            tmp = lindat.split()
            data_dic["year"].append(int(tmp[0]))
            data_dic["DayN"].append(int(tmp[1]))
            data_dic["Hour"].append(int(tmp[2]))
            data_dic["Minute"].append(int(tmp[3]))
            data_dic["data"].append(float(tmp[index]))
            try:
                tme0 = dt.datetime(int(tmp[0]),1,1, int(tmp[2]), int(tmp[3]), 0) + dt.timedelta(days= int(int(tmp[1])-1))
                #print('It is a data with the resolution of minute')
            except:
                tme0 = dt.datetime(int(tmp[0]),1,1, int(tmp[2]), 0, 0) + dt.timedelta(days= int(int(tmp[1])-1))
                #print('It is a hourly data')
            data_dic["time"].append(tme0);

    return data_dic

def plot_2D_mean_std_Model(ModV, ali, tmeids, xax, yax, tiDst, Modname, rdir, alt):
    """
    Plotting the mean and standard deviation as a function of latitude and solar time
    """
    ModMean = np.mean(ModV[:,:,ali,tmeids], axis=2)
    Modstd = np.std(ModV[:,:,ali,tmeids], axis=2)
    StatDen=[ModMean, 3*Modstd]
    # Create a canves to plot our data on.
    fig, axs = plt.subplots(2, figsize=(9, 18), sharex=True)
    
    for ik in range(2):
        cs = axs[ik].contourf(xax, yax, StatDen[ik].T)
        if ik==0:
            axs[ik].set_title('Mean density of '+Modname+' at '+ str(alt)+' km, t = '+tiDst.isoformat(), fontsize=12)
        else:
            axs[ik].set_title('Standard density of '+Modname+' at '+ str(alt)+' km, t = '+tiDst.isoformat(), fontsize=12)
        axs[ik].set_ylabel("Latitudes", fontsize=18)
        axs[ik].tick_params(axis='both', which='major', labelsize=16)
        
        #Make a colorbar for the Contour call
        cbar = fig.colorbar(cs, ax=axs[ik])
        cbar.ax.set_ylabel('Density')
        
    axs[ik].set_xlabel('Local Solar Time', fontsize=18)
    print('Saving the plot to:'+rdir[:-4]+'_Statistics_2D.png')
    #plt.savefig(rdir+'TEC_'+fln[-18:-10]+'.png')
    plt.savefig(rdir[:-4]+'_Statistics_2D.png')
    return


def plot_2D_ModelV(ModV1, ModV2, ali, tmeids, xax, yax, tiDst, Modname, rdir, alt):
    """
    Plotting the two (different) model values as a function of latitude and solar time
    """
    # Create a canves to plot our data on.
    fig, axs = plt.subplots(2, figsize=(9, 18), sharex=True)
    print(np.mean(ModV1[:,:,ali[0],tmeids[0]], axis=2).squeeze().T.shape)
    print(np.mean(ModV2[:,:,ali[1],tmeids[1]], axis=2).squeeze().T.shape)
    for ik in range(2):
        if ik==0:
            cs = axs[ik].contourf(xax[ik], yax[ik], np.mean(ModV1[:,:,0,:], axis=2).squeeze().T)
            axs[ik].set_title('Density of '+Modname[ik]+' at '+ str(alt)+' km, t = '+tiDst[ik].isoformat(), fontsize=12)
        else:
            cs = axs[ik].contourf(xax[ik], yax[ik], np.mean(ModV2[:,:,0,:], axis=2).squeeze().T)
            axs[ik].set_title('Density of '+Modname[ik]+' at '+ str(alt)+' km, t = '+tiDst[ik].isoformat(), fontsize=12)
        axs[ik].set_ylabel("Latitudes", fontsize=18)
        axs[ik].tick_params(axis='both', which='major', labelsize=16)
        
        #Make a colorbar for the Contour call
        cbar = fig.colorbar(cs, ax=axs[ik])
        cbar.ax.set_ylabel('Density')
        
    axs[ik].set_xlabel('Local Solar Time', fontsize=18)
    print('Saving the plot to:'+rdir[:-4]+'_Density_2D.png')
    #plt.savefig(rdir+'TEC_'+fln[-18:-10]+'.png')
    plt.savefig(rdir[:-4]+'_Density_2D.png')
    return
#rdir = "/Volumes/Data/Coding/SpaceWeather/CU2022/Space-Weather-Simulation-Summer-School/day_2/omni_min_def_vglwxZeK03.lst";
rdir = '/Volumes/Data/Data/Space_Weather/Day5/omni2_RefbbJkRPL_Dst.lst'

index = -1;
data_dic = Omni_data_read(rdir, index);

Mxid = np.argmax(data_dic['data']);
Mnid = np.argmin(data_dic['data']);

Adat = np.array(data_dic['data']);
Atme = np.array(data_dic['time']);
L100ID = Adat<-100;
LwTme = data_dic['time'][Mnid];

plt.figure(figsize=(10,8)),
plt.axvline(x=data_dic['time'][Mxid], c='r', linewidth=4, label='Maximum', alpha=0.5);
plt.axvline(x=data_dic['time'][Mnid], c='b', linewidth=4, label='Minimum', alpha=0.5);
plt.plot(data_dic['time'], data_dic['data'], marker='.', c='gray',linewidth=4, label='All Events', alpha=0.5);
plt.plot(Atme[L100ID], Adat[L100ID], '.g', label='<-100 nT', alpha=0.5);
plt.xlabel('Time (hours)'); plt.xlim([data_dic['time'][Mnid]-dt.timedelta(days=1), data_dic['time'][Mnid]+dt.timedelta(days=1)])
plt.ylabel('SYMH (nT)');
plt.title(data_dic['time'][Mnid].isoformat(), fontsize=18 );
plt.grid(True);
plt.legend();

outfile = 'Omni_DST_Sep2002.png'
print('Saving the plot of omni-SymH to:'+rdir[:38]+outfile)
plt.savefig(rdir[:38]+outfile);


del Adat; del Atme; del data_dic; del index; del Mnid; del Mxid; del outfile; del rdir;
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
Rearranging the 2D DATA into 4D data
Let's visualize the density field for 400 KM at given time range (day of the minimum, and also before and after that day) by minimum density.
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
H1time_array_JB2008 = np.linspace(0,8759,8759, dtype = int)
TmeIDs2D = H1time_array_JB2008[250*24:252*24];

# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008,8760), order='F') # Fortran-like index order

#Look for data that correspond to an altitude of 400 km
alt = 400;
hi = np.max(np.where(altitudes_JB2008<=alt)[0])
Modname = 'JB2008'
plot_2D_mean_std_Model(JB2008_dens_reshaped, hi, TmeIDs2D, localSolarTimes_JB2008, latitudes_JB2008, LwTme, Modname, dir_density_Jb2008, alt)


#%%
"""
Rearranging the 2D DATA into 4D data
"""

# First identidy the time index that corresponds to  February 1st, 2002. Note the data is generated at an hourly interval from 00:00 January 1st, 2002

dens_data_feb1 = JB2008_dens_reshaped[:,:,:,TmeIDs2D]

print(dens_data_feb1.shape)

DenPrfP_mean = np.zeros([len(altitudes_JB2008)]);
DenPrfP_std = np.zeros([len(altitudes_JB2008)]);
for ri in range(len(altitudes_JB2008)):
    DenPrfP_mean[ri] = np.mean(dens_data_feb1[:,:,ri,:])
    DenPrfP_std[ri] = np.std(dens_data_feb1[:,:,ri,:])

#%%
"""
Data Visualization II

Now, let's us work with density data from TIE-GCM instead, and plot the density field at given same height as of JB2008
"""
# Import required packages
import h5py
h5dir = '/Volumes/Data/Coding/SpaceWeather/CU2022/TIEGCM/2002_TIEGCM_density.mat'
Ldat = h5py.File(h5dir);
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
H1time_GCM = np.linspace(0,8759,8759, dtype = int)

TmeIDs2D_gcm = H1time_GCM[(int(LwTme.strftime('%j'))-1)*24:(int(LwTme.strftime('%j'))+1)*24];

# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
Dens_GCM_M = np.reshape(Dens_GCM,(NoLst,NoLat,NoAlt,8760), order='F') # Fortran-like index order

#Look for data that correspond to an altitude of 400 km
higcm = np.max(np.where(Alt_GCM<=alt)[0])
Modname = 'TIE-GCM'
plot_2D_mean_std_Model(Dens_GCM_M, higcm, TmeIDs2D_gcm, Solar_tme_GCM, Lat_GCM, LwTme, Modname, h5dir, alt)



#%%
"""
Plotting the Mean densities from both models along wioth standard deviation
"""

# First identidy the time index that corresponds to  February 1st, 2002. Note the data is generated at an hourly interval from 00:00 January 1st, 2002
dens_data_feb1_GCM = Dens_GCM_M[:,:,:,TmeIDs2D_gcm]

print(dens_data_feb1_GCM.shape)

DenPrfP_GCM_mean = np.zeros([len(Alt_GCM)]);
DenPrfP_GCM_std = np.zeros([len(Alt_GCM)]);
for ri in range(len(Alt_GCM)):
    DenPrfP_GCM_mean[ri] = np.mean(dens_data_feb1_GCM[:,:,ri,:])
    DenPrfP_GCM_std[ri] = np.std(dens_data_feb1_GCM[:,:,ri,:])

fig, axs = plt.subplots(1, figsize=(9, 18), sharex=True)
axs.errorbar(DenPrfP_mean, altitudes_JB2008, yerr=3*DenPrfP_std, marker='s', c='blue', label = 'JB2008');
axs.errorbar(DenPrfP_GCM_mean, Alt_GCM, yerr=DenPrfP_GCM_std, c='red', label = 'TIE-GCM');
axs.set_xscale('log'); axs.grid(True);
axs.set_title('Density of JB2008 & TIE-GCM around t = '+LwTme.isoformat(), fontsize=12)
axs.set_xlabel(r"$Density (m^{-3})$", fontsize=18)
axs.set_ylabel("Altitude (km)", fontsize=18)
plt.legend()
print('Saving the plot to:'+h5dir[:-4]+'_Statistics_1D.png')
#plt.savefig(rdir+'TEC_'+fln[-18:-10]+'.png')
plt.savefig(h5dir[:-4]+'_Statistics_1D.png')


plot_2D_ModelV(JB2008_dens_reshaped, Dens_GCM_M, [hi, higcm], [TmeIDs2D, TmeIDs2D_gcm], [localSolarTimes_JB2008, Solar_tme_GCM], [latitudes_JB2008, Lat_GCM], [LwTme, LwTme], ['JB2008', 'TIE-GCM'], dir_density_Jb2008, alt)

#%%

