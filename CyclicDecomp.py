"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
Cyclic Decompositions for Preference Lists
"""

import numpy as np
from itertools import permutations
from functools import reduce
import matplotlib.pyplot as plt
import scipy as sc

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
    def print_by_value(self, type = "fourier", hspace = 20, hide_zero = True, precision = 6):
        """
        Prints the type of coefficients desired by thier value.
        """
        types = ["fourier", "stat", "basis", "basis_inv"]
        assert type in types
        if type == types[0]:
            interesting = self.fourier
        elif type == types[1]: 
            interesting = self.stat
        elif type == types[2]:
            interesting, labels = self.new_basis()
        elif type == types[3]:
            interesting, labels = self.new_basis(True)
        vals = np.unique(np.around(np.reshape(interesting, self.n**self.m),precision), axis = 0)

        total = 0
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
            if type in types[2:]:
                for i in range(len(close)):
                    if close[i]:
                        print(labels[i], end = ", ")
                        total += 1
            else:
                for idx, x in np.ndenumerate(interesting):
                    if close[idx]:                    
                        print(str(tuple(reversed(idx))), end = "")
                        total += 1
            print()
        print(total)

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
        pointwise product in fourier domain (leaves out division...)
        input: another cycle decomposition object
        returns: the pointwise product of the two objects            
        """
        assert self.n == other.n
        assert self.m == other.m
        return CnmStat(self.n, self.m, self.fourier * other.fourier, "fourier")

    def expected_val(self, other):
        """
        Gives expected value for distribution and statistic
        """
        assert self.n == other.n
        assert self.m == other.m
        return np.sum(self.stat * other.stat)/(self.n**self.m)

    def distribution(self):
        """
        Turns a cyclic decomposition obejct into a probability distribution 
        on the fourier coefficients
        """
        to_add = np.min(self.stat)
        if to_add < 0:
            self.stat = self.stat - to_add
            self.fourier[[1]*self.m] -= to_add

        norm = np.sum(self.stat)
        if norm != 0:
            self.stat = self.stat/norm
            self.fourier = self.fourier/norm

    def average_Sm(self):
        """
        averages over orbits of the symmetric group
        """
        self.average(self.get_Sm_orbits())

    def get_Sm_orbits(self):
        """
        Returns a partition of the indices by the orbits of $S_m$
        """
        ind = np.arange(self.n**self.m)
        part = []
        while len(ind) != 0:
            part.append([])
            
            s = np.base_repr(ind[0], self.n)
            s = "0"*(self.m - len(s)) + s

            for sigma in permutations(s):
                part[-1].append(int("".join(sigma), self.n))
            part[-1] = np.unique(part[-1])
            ind = np.setdiff1d(ind, part[-1])
        return part
    
    def average(self, partition):
        """
        averages over the partition given
        """
        arr = np.reshape(self.stat, self.n**self.m).astype(float)
        for part in partition:
            ave = sum(arr[part])/len(part)
            arr[part] = np.ones(len(part)) * ave
        self.set_stat(arr)

    def new_basis_strs(self, inv = False):
        """
        generates a list of n^m sets containing indices for the corresponding sets
        """
        ind = np.arange(self.n**self.m) # indices - will be exhausted by end
        part = []                       # the parts that will be returned
        strs = []                       # readable strings labling parts
        while len(ind) != 0:
            arr = []    # array for orbits of $S_n$ acting on the string
            
            s = np.base_repr(ind[0], self.n)
            s = "0"*(self.m - len(s)) + s
            if inv:
                s = s[::-1]
            
            if inv:
                perm_strings = np.array(["".join(x)[::-1] for x in permutations(s)])
            else:
                perm_strings = np.array(["".join(x) for x in permutations(s)])
            used_ind = [int(x, self.n) for x in np.unique(perm_strings)]
            ind = np.setdiff1d(ind, used_ind) # remove used strings from ind

            arr.append([perm_strings])
            # set up of the ith index of arr is the set of orbit of $S_{n-i}$
            for i in range(1,self.m):
                # The number of pemutations is (m - i)!
                step = np.math.factorial(self.m - i)
                arr.append(np.split(perm_strings, np.arange(step, len(perm_strings), step), axis = 0))

            # turns each orbit into a set
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    arr[i][j] = set(np.unique(arr[i][j]))

            # eliminates duplicates/unnecessary orbits by iterating through and setting them to set()
            for i in range(len(arr)):
                unique = np.unique(arr[i], True)[1]
                for j in range(len(arr[i])):
                    if j not in unique or len(arr[i][j]) == 1 and arr[i][j] != set():
                        for k in range(1, len(arr) - i):
                            step = 1 
                            # i + kth row : len/(self.m - (i + k))!
                            # ith row     : (len/(self.m - i)!) 
                            # step size   : (self.m - i)! /(self.m - (i + k))!
                            for l in range(k):
                                step *= len(arr) - (i + l)
                            for l in range(j*step, (j+1)*step):
                                arr[i + k][l] = set()
                    if j not in unique:
                        arr[i][j] = set()
                    
            # iterates through and adds things to the answer to be returned
            # removes the extra orbits that would make things linearly dependant
            strs.append("S_" + str(self.m) + "*" + (str(next(iter(arr[0][0])))[::-1]))
            part.append(arr[0][0])
            for i in range(1, len(arr)):
                # ith row has (len/(self.m - i)!) 
                # i+1th row has (len/(self.m - i - 1)!) 
                step = len(arr) + 1 - i
                arr2 = np.split(arr[i], np.arange(step, len(arr[i]), step), axis = 0)
                for j in range(len(arr2)):
                    arr2[j] = list(filter(lambda x: x != set(), arr2[j]))
                    if arr2[j] == []:
                        continue
                    else:
                        # picks minind to be removed so that the set is linearly independent
                        minind = 0
                        for k in range(len(arr2[j])):
                            if inv:
                                # rearranging end of string 
                                # so beginning of the string matters/is fixed
                                if not next(iter(arr2[j][k]))[::-1] < next(iter(arr2[j][minind]))[::-1]:
                                    minind = k 
                            else:
                                # rearranging first cars. Result will still be reversed.
                                # so beginning of the string matters/is fixed    
                                # throws out latest alphabetically from end
                                if not next(iter(arr2[j][k])) < next(iter(arr2[j][minind])):
                                    minind = k 
                                
                            # old version from 7/10
                            #if len(arr2[j][k]) < len(arr2[j][minind]):
                            #    minind = k
                            #elif len(arr2[j][k]) == len(arr2[j][minind]):
                            #    if next(iter(arr2[j][k])) < next(iter(arr2[j][minind])):
                            #        minind = k 
                        # insert into answer which will be returned
                        for k in range(len(arr2[j])):
                            if k != minind:
                                strs.append("S_" + str(self.m - i) + "*" + (str(next(iter(arr2[j][k])))[::-1]))
                                part.append(arr2[j][k])       
        return part, strs

    def new_basis(self, inv = False):
        """
        computes the statistic in the modified basis and prints the result
        """
        basis, s = self.new_basis_strs(inv)

        for i in range(len(basis)):
            basis[i] = [int(s, self.n) for s in basis[i]]

        arr = np.zeros((self.n**self.m, self.n**self.m), int)
        for i in range(len(basis)):
            for b in basis[i]:
                arr[b][i] = 1

        from scipy.sparse import csr_array
        
        change_basis_matrix = sc.sparse.csr_array(np.linalg.inv(arr))
        stat_in_new_basis = change_basis_matrix.dot(np.reshape(self.fourier, self.n**self.m))

        return stat_in_new_basis, s

    def sample(self):
        """
        Turns the object into a distribution, and then samples from that distribution
        """
        self.distribution()
        vals = [ idx for idx,x in np.ndenumerate(self.stat)]
        index = np.random.choice(np.arange(len(vals)), p=np.reshape(self.stat, self.n**self.m))
        return vals[index] 

    def plot(self):
        """
        Plots a histogram of the values of the statistic
        """
        plt.hist(np.reshape(self.stat, self.n**self.m))
        plt.show()
    
    ##################
    # Static Methods #
    ##################

    @staticmethod
    def cos(n, m = None, freq = None, phase = None):
        """
        freq - an array of frequencies for the sin waves
        """
        if m == None:
            if type(freq) != type(None):
                m = len(freq)

        if type(freq) == type(None):
            freq = np.ones(m, int)

        if type(phase) == type(None):
            phase = np.zeros(m)
        
        def one_cos(f,p):
            return 1 + np.cos(np.arange(0, f*2*np.pi, f*2*np.pi/n)+ np.pi*p)
            """
            L = np.zeros(n, complex)
            L[0] = 1
            L[f] += np.exp(p * 1j) * .5
            L[-f] += np.exp(p * -1j) * .5
            return L
            """
        
        ans = one_cos(freq[0], phase[0])
        for i in range(1, m):
            ans = np.kron(ans, one_cos(freq[i], phase[i]))

        s = CnmStat(n,m,ans,"stat")
        s.distribution()
        return s
            
