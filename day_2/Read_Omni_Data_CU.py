#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:48:19 2022

@author: selva
"""
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
            tme0 = dt.datetime(int(tmp[0]),1,1, int(tmp[2]), int(tmp[3]), 0) + dt.timedelta(days= int(int(tmp[1])-1))
            data_dic["time"].append(tme0);

    return data_dic

rdir = "/Volumes/Data/Coding/SpaceWeather/CU2022/Space-Weather-Simulation-Summer-School/day_2/omni_min_def_vglwxZeK03.lst";

index = -1;
data_dic = Omni_data_read(rdir, index);

Mxid = np.argmax(data_dic['data']);
Mnid = np.argmin(data_dic['data']);

Adat = np.array(data_dic['data']);
Atme = np.array(data_dic['time']);
L100ID = Adat<-100;

plt.figure(figsize=(10,8)),
plt.axvline(x=data_dic['time'][Mxid], c='r', label='Maximum', alpha=0.5);
plt.axvline(x=data_dic['time'][Mnid], c='b', label='Minimum', alpha=0.5);
plt.plot(data_dic['time'], data_dic['data'], marker='.', c='gray', label='All Events', alpha=0.5);
plt.plot(Atme[L100ID], Adat[L100ID], marker='.', c='green', label='<-100 nT', alpha=0.5);
plt.xlabel('Year of 2013');
plt.ylabel('SYMH (nT)');
plt.grid(True);
plt.legend();

outfile = 'Omni_SymH.png'
print('Saving the plot of omni-SymH to:'+rdir[:-33]+outfile)
plt.savefig(rdir[:-33]+outfile)

