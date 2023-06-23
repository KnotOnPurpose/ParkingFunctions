"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
Cyclic Decompositions for Preference Lists
"""

import numpy as np

"""
TODO 
add in modified basis
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
        self.DFT = CnmStat.get_dft(self.n, self.m)
        assert arr_type in ["stat", "fourier"]

        # Varibles which store calculated information for common statistics
        self.disp_i = None # stores counts of displacement exactly i for circular data
        self.total_disp = None

        # Set up variables for details of statistic
        if type(arr) == type(None):
            self.stat = np.zeros(self.n**self.m)
            self.fourier = np.zeros(self.n**self.m)
        else:
            if arr_type == "fourier":
                self.set_fourier(np.array(arr))
            else:
                self.set_stat(np.array(arr))
        
        assert len(self.stat) == self.n**self.m
        assert len(self.fourier) == self.n**self.m

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
        self.stat = arr
        self.fourier = self.DFT.dot(self.stat)

    def set_fourier(self, arr):
        """
        input: 
            arr - the new value for the fourier coefficients
        result:
            sets the fourier coefficients to the values of arr
            updates all other dependant arrays
        """
        self.fourier = arr
        self.stat = np.matrix(self.DFT).getH().dot(self.fourier)/(self.n**self.m)

    ###################
    # Display options #
    ###################
    def print_by_value(self, t = "fourier", hspace = 20):
        """
        """
        if t == "fourier":
            interesting = self.fourier
        else: 
            interesting = self.stat
        vals = np.unique(np.around(interesting,6), axis = 0)

        print(len(vals) - 1)
        for j in range(len(vals)):
            if np.prod(vals[j] == 0):
                continue

            if len(np.shape(interesting)) == 1:
                close = np.isclose(interesting, [vals[j]] * len(interesting))
            else:
                close = np.isclose(interesting, np.ones(np.shape(interesting)) * [vals[j]])
                close = np.prod(close, axis = 1)
            
            s = np.array2string(vals[j], precision=3)
            s += " "*(hspace - len(s))
            print(s + " : ", end = "")
            for i in range(len(interesting)):
                if close[i]:
                    print(self.get_group_str(i), ",", sep = "", end = " ")
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

    ####################
    # HELPER FUNCTIONS #
    ####################

    def get_group_str(self, i):
        """
        Given an integer i, returns the subscript of the corresponding character
        """
        index = str(np.base_repr(i, self.n))
        index = "0" * (self.m-len(index)) + index
        index = index[::-1]
        return index

    @staticmethod
    def get_dft(n,m = None):
        """
        Inputs:
            n - the number of parking spots
            m - the number of cars to park
        Outputs:
            returns the dft for $C_n^m$ 
        """
        m = m or n

        DFTn = np.fft.fft(np.eye(n))
        if m == 1:
            return DFTn
        else:
            DFT = np.kron(DFTn, DFTn)
            for i in range(m - 2):
                DFT = np.kron(DFT, DFTn)
            return DFT
