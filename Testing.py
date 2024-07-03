"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
This is a file for writing bad code quickly to find answers. 
Worry about making things modular later - just write code.
"""

import sympy as sp

from Parking import *
from Plotting import * 
from CyclicDecomp import *
from matplotlib.widgets import Slider, Button

from TestData import * 
np.set_printoptions(suppress=True, precision=4, linewidth = 1000)

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

###########
# June 22 #
###########
def test_convolve():
    sin = CnmStat(7,7,np.kron(np.kron(np.kron(np.kron(np.kron(np.kron([1,.5,0,0,0,0,.5], [1/7]*7), [1/7]*7),[1/7]*7),[1/7]*7),[1/7]*7),[1/7]*7), "fourier")
    
    stats = IterateStats(7,7)
    stats.iterate()
    
    test = CnmStat(7,7)
    test.set_stat(stats.total_disp)
    
    return test.convolve(sin)

###########
# June 27 #
###########

def random_walk(n,m,circular = True, trials = 1000, starting_distribution = None, steps = 100):
    """
    inputs:
        n - number of spots
        m - number of cars
        circular - circular parking or not
        trials - number of trials to run
        starting_distribution - a function which takes in n,m,circular and returns a preference list
        steps - number of steps of random walk to take
    """
    if starting_distribution == None:
        starting_distribution = lambda m,n,circular: Park(np.ones(m, int), n, circular = circular)
    
    parking_res = [None]*trials
    for i in range(trials):
        park = starting_distribution(m,n,circular)
        park.walk(steps)
        parking_res[i] = park
    return parking_res

def plot_stat(parking, stat):
    """
    Given a list of preference lists, plots a histogram of the statistic of interst
    inputs:
        parking - array of Park objects
        stat - statistic of interest
    """
    if stat == None:
        stat = lambda park: park.defect()
    data = list(map(stat, parking))
    plt.hist(data, bins = np.arange(-0.5, max(data)+0.6, 1))
    plt.show()

def get_ind(n,m,c):
    """
    Gets an indicator function which is 1 on prefernce lists where the ith car is lucky
    """
    ind = []
    park = Park([1] * m, n, circular = True)
    for i in range(n**m):
        if park.displacement[c] == 0:
            ind.append(1)
        else:
            ind.append(0)
        park.next()
    return ind

##########
# July 3 #
##########

def random_sin_walk():
    """
    inputs:
        n - number of spots
        m - number of cars
        circular - circular parking or not
        trials - number of trials to run
        starting_distribution - a function which takes in n,m,circular and returns a preference list
        steps - number of steps of random walk to take
    """
    n = 6
    m = 6
    circular = True
    t_max = 10

    dat = np.zeros(n**m)
    dat[0] = 1
    d = CnmStat(n,m, dat, arr_type="stat")
    w = CnmStat.cos(n,m)
    stats = IterateStats(n,m, circular)
    s = CnmStat(n,m, stats.disp_i[0])
    
    data = [d.expected_val(s)]
    for i in range(t_max):
        d = d.convolve(w)
        print(np.sum(d.stat))
        data.append(d.expected_val(s))
        
    print(data)

    plt.plot(data)
    plt.show()

def random_sin_walk_interactive():
    """
    inputs:
        n - number of spots
        m - number of cars
        circular - circular parking or not
        trials - number of trials to run
        starting_distribution - a function which takes in n,m,circular and returns a preference list
        steps - number of steps of random walk to take
    """
    n = 6
    m = 6
    circular = True
    
    t_max = 10

    dat = np.zeros(n**m)
    dat[0] = 1
    d = CnmStat(n,m, dat, arr_type="stat")
    w = CnmStat.cos(n,m)
    stats = IterateStats(n,m, circular)
    s = CnmStat(n,m, stats.disp_i[0])

    # The parametrized function to be plotted
    def f(x, steps):
        dwt = CnmStat(n,m)
        dwt.set_fourier(d.fourier * (w.fourier**int(steps)))
    
        ans = np.zeros(len(x))
        for i in range(len(ans)):
            ans[i] = sum(dwt.stat[s.stat == x[i]])
        return ans

    x = np.arange(0, n + 1, 1, int)

    # Define initial parameters
    # https://matplotlib.org/stable/gallery/widgets/slider_demo.html
    init_steps = 0

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    line, = ax.plot(x, f(x, init_steps), lw = 1)
    ax.set_xlabel('Statistic')

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frequency.
    # axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    # req_slider = Slider(
    #    ax=axfreq,
    #    label='Frequency [Hz]',
    #    valmin=0.1,
    #    valmax=30,
    #    valinit=init_frequency,
    #)

    # Make a vertically oriented slider to control the amplitude
    axsteps = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    steps_slider = Slider(
        ax=axsteps,
        label="Amplitude",
        valmin=0,
        valmax=20,
        valinit=init_steps,
        valstep=1,
        orientation="vertical"
    )


    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_ydata(f(x, steps_slider.val))
        fig.canvas.draw_idle()


    # register the update function with each slider
    #freq_slider.on_changed(update)
    steps_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    def reset(event):
        steps_slider.reset()
    button.on_clicked(reset)

    plt.show()
    
def indicator(n,m,i,d, circular = True):
    """
    indicator function for the ith car was displaced $d$
    """
    stat = np.zeros(n**m, int)
    park = Park([1] * m,n, circular)

    for ind in range(n**m):
        if park.displacement[i - 1] == d:
            stat[ind] = 1
        park.next()
    return stat

def spot_indicator(n,m,i,d, circular = True):
    #TODO currently just a copy of indicator
    stat = np.zeros(n**m, int)
    park = Park([1] * m,n, circular)

    for ind in range(n**m):
        if park.displacement[i - 1] == d:
            stat[ind] = 1
        park.next()
    return stat

def orbits_involved(stat):
    """
    I have a hunger to know how things play out for larger than 5 for things constant 
    under reordering
    """
    u = np.unique(np.around(np.reshape(stat.fourier, stat.n**stat.m),6))
    d = {}
    for idx, x in np.ndenumerate(stat.fourier):
        difference_array = np.absolute(u-x)
        index = difference_array.argmin()

        if u[index] not in d:
            d[u[index]] = set()
        
        s = list(idx)
        s.sort()
        d[u[index]].add(tuple(s))

    return d

