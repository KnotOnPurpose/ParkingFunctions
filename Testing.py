"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
This is a file for writing bad code quickly to find answers. 
Worry about making things modular later - just write code.
"""


from Parking import *
from TestData import * 
import sympy as sp
from Plotting import * 

np.set_printoptions(suppress=True, precision=4)

"""
#TODO 
A theme has emerged in that I decomposed functions a Lot. 
It would be good to make a differnt file that is just function decomposition 
instead of lumping everything into the testing file.
"""

"""
June 9th ish
"""
def test_circular(n, m):
    d = {}
    park = Park([1] * m, n, circular = True)

    for i in range(n**m):
        filling = tuple(park.lot)
        if filling in d:
            d[filling] += 1
        else:
            d[filling] = 1
        park.next()

    return d

def test_parking_func(n,m):
    """
    Truely the epitome of inefficent
    """
    d = {}
    park = Park([1] * m, n, circular = False)

    for i in range(n**m):
        if(park.defect() == 0):
            filling = tuple(park.lot)
            if filling in d:
                d[filling] += 1
            else:
                d[filling] = 1
        park.next()

    return d

def print_by_conjugacy_class():
    #d_1 = test_circular(4, 4)
    d = test_parking_func(4, 4)

    print("Partition = 1,1,1,1")
    print(d[(1,2,3,4)])
    print()
    print("Partition = 2,1,1")
    print(d[(2,1,3,4)], d[(1,3,2,4)], d[(1,2,4,3)], d[(4,2,3,1)])
    print(d[(3,2,1,4)], d[(1,4,3,2)])
    print()
    print("Partition = 3,1")
    print(d[(3,1,2,4)], d[1,4,2,3], d[(4,2,1,3)], d[(4,1,3,2)])
    print(d[(2,3,1,4)], d[1,3,4,2], d[(3,2,4,1)], d[(2,4,3,1)])
    print()
    print("Partition = 2,2")
    print(d[(2,1,4,3)], d[(4,3,2,1)], d[(3,4,1,2)])
    print()
    print("Partition: 4")
    print(d[(2,3,4,1)], d[(4,1,2,3)])
    print(d[(2,4,1,3)], d[(3,1,4,2)])
    print(d[(3,4,2,1)], d[(4,3,1,2)])

def print_by_cosets_of_cyclic_group():
    #d = test_circular(4, 4)
    d = test_parking_func(4, 4)

    print("coset representative: 1,2,3,4. ")
    print("e, (1234), (13)(24), (1432)")
    print(d[(1,2,3,4)], d[(2,3,4,1)], d[(3,4,1,2)], d[(4,1,2,3)])
    print()
    print("coset representative: 1,2,4,3")
    print("(34), (124), (1423), (132)")
    print(d[(1,2,4,3)], d[(2,4,3,1)], d[(4,3,1,2)], d[(3,1,2,4)])
    print()
    print("coset representative: 1,3,2,4")
    print("(23), (134), (1243), (142)")
    print(d[(1,3,2,4)], d[(3,2,4,1)], d[(2,4,1,3)], d[(4,1,3,2)])
    print()
    print("coset representative: 1,3,4,2")
    print("(234), (1324), (143), (12)")
    print(d[(1,3,4,2)], d[(3,4,2,1)], d[(4,2,1,3)], d[(2,1,3,4)])
    print()
    print("coset representative: 1,4,2,3")
    print("(243), (14), (123), (1342)")
    print(d[(1,4,2,3)], d[(4,2,3,1)], d[(2,3,1,4)], d[(3,1,4,2)])
    print()
    print("coset representative: 1,4,3,2")
    print("(24), (14)(23), (13), (12)(34)")
    print(d[(1,4,3,2)], d[(4,3,2,1)], d[(3,2,1,4)], d[(2,1,4,3)])
    
"""
June 12th
"""

def decompose(stat , n, m = None, print_output = True):
    """
    inputs:
        stat - the statistic to be decomposed
        n - the number of parking spots
    result: decomposes the function into the character basis for $C_m^n$
    """
    m = m or n

    DFTn = np.fft.fft(np.eye(n))
    DFT = np.kron(DFTn, DFTn)
    for i in range(m - 2):
        DFT = np.kron(DFT, DFTn)

    ans = DFT.dot(stat)
    if print_output:
        close = np.isclose(ans, [0] * len(ans))
        for i in range(len(ans)):
            if not close[i]:
                index = str(np.base_repr(i, n))
                index = "0"*(m - len(index)) + index
                print(index[::-1], " : ", ans[i])
    return ans
    
def get_disp(n,m):
    """
    Inputs:
        n - number of spots
        m - number of cars
    Returns an m by n**m long array filled with counts
    """
    ans = np.zeros((m, n**m), int)
    park = Park([1] * m, n, circular = True)

    for i in range(n**m):
        t = 0
        for j in range(m):
            cnt = park.displacement.count(j)
            ans[j][i] = cnt
            t += cnt
            if cnt == m:
                break
        
        park.next()

    return ans

def total_disp(n,m):
    """
    Inputs:
        n - number of spots
        m - number of cars
    returns an n**m long array filled with counts
    """
    ans = np.zeros(n**m, int)
    park = Park([1] * m, n, circular = True)

    for i in range(n**m):
        t = 0
        ans[i] = park.displacement_total()      
        park.next()

    return ans
    #return np.sum(np.transpose(get_disp(n,m)) * list(range(m)), axis = 1)

def by_value(interesting, n, m, hspace = 20, labels = None):
    """
    Prints subscript of decomposition based on fourier transform output values
    """
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
                index = ""
                if labels == None:
                    index = str(np.base_repr(i, n))
                    index = "0" * (m-len(index)) + index
                    index = index[::-1]
                else:
                    index = labels[i] 
                print(index, ",", sep = "", end = " ")
        print()

def group_characters(n,m):
    """
    Code written to collect all of the values of interest together to be able 
    to see what characters are always grouped with eachother
    """
    counts = get_disp(n,m)
    fourier_coeff = np.zeros((len(counts), n**m), complex)

    for i in range(len(counts)):
        fourier_coeff[i] = decompose(counts[i], n, m, False)
    
    return np.transpose(fourier_coeff)

def build_new_basis(n, m):
    """
    Input: n,m
    Output: an n**m by n**m matrix which represents the change of basis 
                        from the new basis back to the original basis
    """


        
