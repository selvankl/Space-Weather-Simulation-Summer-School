#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Selvaraj'
__email__ = 'selvankl@ymail.com'

from math import factorial
from math import pi


def cos_approx(x, accuracy=20):
    """
    Cosine of x is approximated by Tylor expansion
    """
    cosxV = sum([(((-1)**ij)/(factorial(2*ij))) * (x**(2*ij)) for ij in range(accuracy)])
        #print(cosxV)
    return cosxV



# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
