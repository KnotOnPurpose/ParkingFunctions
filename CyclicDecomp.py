"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
Cyclic Decompositions for Preference Lists
"""

import numpy as np

"""
TODO 
Add in modified basis - figure out theoretical framework for constructing modified basis
"""

class CnmStat:
    """
    Class variables
    n - number of spots
    m - number of cars
    DFT - discrete fourier transform for C_n^m
    stat - the statistic of interest as a list of elements in kronicer product order
    fourier - the fourier transform of the statistic
    """
    def __init__(self, n, m = None, arr = None, arr_type = "stat"):
        """
        Inputs:
            n - number of spots
            m - number of cars
            arr - an array 
            arr_type - the type of the array. 
                Should be one of the following: "stat", "fourier"
        """
        # Set up parameter type variables
        self.n = n
        self.m = m or n
        self.shape = tuple([self.n]*self.m)
        #self.DFT = CnmStat.get_dft(self.n, self.m)
        assert arr_type in ["stat", "fourier"]

        # Varibles which store calculated information for common statistics
        self.disp_i = None # stores counts of displacement exactly i for circular data
        self.total_disp = None

        # Set up variables for details of statistic
        if type(arr) == type(None):
            self.set_stat(np.zeros(self.n**self.m))
        else:
            if arr_type == "fourier":
                self.set_fourier(arr)
            else:
                self.set_stat(arr)

    ########################
    # change/set statistic #
    ########################
    def set_stat(self, arr):
        """
        input: 
            arr - the new value for the stats array
        result:
            sets the stats to the values of arr
            updates all other dependant arrays
        """
        self.stat = np.reshape(arr, self.shape)
        self.fourier = np.fft.fftn(self.stat)

    def set_fourier(self, arr):
        """
        input: 
            arr - the new value for the fourier coefficients
        result:
            sets the fourier coefficients to the values of arr
            updates all other dependant arrays
        """
        self.fourier = np.reshape(arr, self.shape)
        self.stat = np.fft.fftn(self.fourier)

    ###################
    # Display options #
    ###################
    def print_by_value(self, t = "fourier", hspace = 20, hide_zero = True, precision = 6):
        """
        """
        if t == "fourier":
            interesting = self.fourier
        else: 
            interesting = self.stat
        vals = np.unique(np.around(np.reshape(interesting, self.n**self.m),precision), axis = 0)

        if hide_zero:
            print(len(vals)-1)
        else:
            print(len(vals))
        for i in range(len(vals)):
            if hide_zero and np.isclose(vals[i],0):
                continue
            
            s = np.array2string(vals[i], precision=precision)
            s += " "*(hspace - len(s))
            print(s + " : ", end = "")

            close = np.isclose(interesting, vals[i], atol = 10**(-precision))
            for idx, x in np.ndenumerate(interesting):
                if close[idx]:
                    print(str(idx), end = "")
            print()

    #####################
    # USEFUL OPERATIONS #
    #####################
    def times(self, other):
        """
        pointwise product
        input: another cycle decomposition object
        returns: the pointwise product of the two objects            
        """
        assert self.n == other.n
        assert self.m == other.m
        return CnmStat(self.n, self.m, self.stat * other.stat, "stat")
    
    def convolve(self, other):
        """
        pointwise product
        input: another cycle decomposition object
        returns: the pointwise product of the two objects            
        """
        assert self.n == other.n
        assert self.m == other.m
        return CnmStat(self.n, self.m, self.fourier * other.fourier, "fourier")
