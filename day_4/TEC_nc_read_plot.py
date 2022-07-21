#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:27:49 2022

@author: selva
"""
__author__ = 'Selvaraj'
__email__ = 'selvankl@ymail.com'

import netCDF4 as nc
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    """
    Plotting the TEC by reading  the netCDF files
    Primarly Needs two types of inputs, one is directory and the second is filename(s)
    The filenames can be placed with one space gap in series
    plot of TEC is saved in the same directory of the data
    """
    parser = argparse.ArgumentParser(description = 'Plotting the TEC by reading netCDF file')
    
    parser.add_argument('rdir',  \
                        help = 'directory ', \
                            type=str)
    parser.add_argument('nc_file', nargs='+',\
                        help = 'Input file name ', \
                            type=str)
    args = parser.parse_args()
    return args

#rdir = '/Volumes/Data/Data/Space_Weather/Day4/WFS/'
#fln = 'wfs.t00z.ipe05.20220719_210500.nc'

def plot_TEC(dataset, fln, vmn=0, vmx=100, figsize=(12, 9)):
    """
    Definition for pcolormesh which plots TEC as function of longitude and latitude
    plot_TEC function takes four necessary inputs (x-longitude, y-latitude, z-TEC, filename for the title) with one optional (figsize)
    """
    #####--identifing the variables from data--
    lat = dat['lat'][:] #latitude
    lng = dat['lon'][:] #longitude
    TEC = dat['tec'][:] #Column integrated total electron density
    latU = dat['lat'].units
    lngU = dat['lon'].units
    tecU = dat['tec'].units

    TEC[TEC==9.969209968386869e+36] = np.nan #fill values are replaced by NAN

    dateID = fln[-18:-10][6:8] +'-'+fln[-18:-10][4:6] +'-'+fln[-18:-10][0:4] #Date to print
    flnM = dateID+'     Column integrated TEC within '+str(int(np.min(dat['alt'][:]))) +' - '+str(int(np.max(dat['alt'][:]))) +' km'
    #--1 TECu = 1e-16 electrons / m3
    ######--
    
    fig, axs = plt.subplots(1, figsize=figsize)
    cs = axs.pcolormesh(lng,lat,TEC, vmin = vmn, vmax=vmx, shading='auto') #plotting with min and max of TEC
    axs.set_title(flnM, fontsize=18) #title
    axs.set_ylabel(f'Latitude ({latU})', fontsize=18) #ylabel
    axs.tick_params(axis='both', which='major', labelsize=16)
    #Make a colorbar for the Contour call
    cbar = fig.colorbar(cs, ax=axs)
    cbar.ax.set_ylabel(tecU, fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    axs.set_xlabel(f'Longitude ({lngU})', fontsize=18)

    return fig, axs


##dates = [datetime(yri, monthi, dayi) for yri, monthi, dayi in zip(yrs, months, days)]

# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    """
    To access the multiple files, for loop is used here
    """
    args = parse_args()
    print('\n The Data Directory: '+args.rdir+'\n')
    print('\n No. of files are given to process: '+str(len(args.nc_file))+'\n')
    for nfi in range(len(args.nc_file)):
        dat = nc.Dataset(args.rdir+'/'+args.nc_file[nfi]) # reading netCDF file into dat
        print(dat) #print the data
    
        # plot_TEC function takes four necessary inputs (x-longitude, y-latitude, z-TEC, filename for the title) with three optional (figsize)
        fig, axs = plot_TEC(dat, args.nc_file[nfi], 0, 100)
        print('Saving the plot to:'+args.rdir+'TEC_'+args.nc_file[nfi][-18:-10])
        #plt.savefig(rdir+'TEC_'+fln[-18:-10]+'.png')
        plt.savefig(args.rdir+'/'+args.nc_file[nfi]+'.png')

