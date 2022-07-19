#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:03:27 2022

@author: selva
"""

#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Selvaraj'
__email__ = 'selvankl@ymail.com'

from math import factorial
from math import pi
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description = 'Cosine of x is approximated by Tylor expansion')
    parser.add_argument('in_var', nargs = 1, \
                        help = 'Input Variable - need two but -npts is optional!', \
                            type=float)

    # npts: scalar value, type integer, default 5:
    parser.add_argument('-npts', \
                       help = 'another scalar (default = 5)', \
                       type = int, default = 10)
    args = parser.parse_args()
    return args

def cos_approx(x, accuracy=10):
    """
    Cosine of x is approximated by Tylor expansion
    """
    cosxV = sum([(((-1)**ij)/(factorial(2*ij))) * (x**(2*ij)) for ij in range(accuracy)])
        #print(cosxV)
    return cosxV


def is_close(value, close_to, eta=1.e-2):
    """Returns True if the value is close to eta, or false otherwise"""
    return value > close_to-eta and value < close_to+eta

# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    
    args = parse_args()
    print(args.in_var)
    print(args.npts)
    
    ax1v = cos_approx(args.in_var[0], args.npts)
    print("The number of points = ", args.npts)
    print("The angle to approximate= ", args.in_var[0])
    print("cos("+str(args.in_var[0])+') = ', ax1v)
    #
    #print("cos(0) = ", cos_approx(0))
    #print("cos(pi) = ", cos_approx(pi))
    #print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
    assmJ = is_close(ax1v, np.cos(args.in_var[0])), "cos("+str(args.in_var[0])+') = '+"  is   "+str(np.cos(args.in_var[0]))
    print(assmJ)
    """
    args = parse_args()
    print(args)
    in_var = args.in_var
    print(in_var)
    # grab the number of points (an integer, default 5):
    npts = args.npts
    print(npts)
    """