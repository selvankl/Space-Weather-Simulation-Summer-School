#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:36:51 2022

@author: selva
"""
__author__ = 'Selvaraj'
__email__ = 'selvankl@ymail.com'


import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import argparse

import locale
locale.getpreferredencoding(False)

def parse_args():
    parser = argparse.ArgumentParser(description = 'Cosine of x is approximated by Tylor expansion')
    parser.add_argument('infle', nargs = 1, \
                        help = 'directory including Input file name - need two input but (-out) output file name is optional!', \
                            type=str)

    parser.add_argument('-index', \
                       help = 'another scalar (default = -1)', \
                       type = int, default = -1)
    parser.add_argument('-out', \
                       help = 'Output file name for the figure (default = out.png)', \
                       type = str, default = 'out.png')
    args = parser.parse_args()
    return args

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


# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    
    args = parse_args()
    print(args.infle)
    print(args.index)
    print(args.out)
    
    data_dic = Omni_data_read(args.infle[0], args.index);
    
    Mxid = np.argmax(data_dic['data']);
    Mnid = np.argmin(data_dic['data']);
    
    Adat = np.array(data_dic['data']);
    Atme = np.array(data_dic['time']);
    L100ID = Adat<-100;
    
    plt.figure(),
    plt.axvline(x=data_dic['time'][Mxid], c='r', label='Maximum', alpha=0.5);
    plt.axvline(x=data_dic['time'][Mnid], c='b', label='Minimum', alpha=0.5);
    plt.plot(data_dic['time'], data_dic['data'], marker='.', c='gray', label='All Events', alpha=0.5);
    plt.plot(Atme[L100ID], Adat[L100ID], marker='.', c='green', label='<-100 nT', alpha=0.5);
    plt.xlabel('Year of 2013');
    plt.ylabel('SYMH (nT)');
    plt.grid(True);
    plt.legend();
    
    #outfile = 'Omni_SymH.png'
    print('Saving the plot of omni-SymH to:'+rdir[:-33]+args.out);
    plt.savefig(args.infle[0][:-33]+args.out);
    print('The minimum of SYMH in nT = '+str(data_dic['data'][Mnid]))
    print('The time at which minimum of SYMH occurs = '+str(data_dic['time'][Mnid].isoformat()))
    
    
    
 
