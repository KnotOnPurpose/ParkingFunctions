"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
Demo
This is a file whose goal is to be a set of demonstrations for the 
more refined pieces of code that I have. (because I find this part of documentation useful)
TODO actually fill in the demos
"""

import numpy as np

from Parking import * 
from Plotting import *
from CyclicDecomp import *

#################
# Parking Demos #
#################

#TODO show how to work with IterateStats
def iterate_stats_demo():
    #initializes statistic - will pull from data if possible, otherwise iterates through
    stats = IterateStats(6,6,False)

    #runs the iterative algorithm and saves the result
    IterateStats.update_file("*filename*")

##############
# Plot Demos #
##############

def plotting_demo_settings():
    plot = Plot()

    # sets color map to use
    plot.cm = plt.cm.Oranges
    
    # sets default figure size
    plot.figsize = (3,7)

    # sets width of the bars. By default 1
    plot.bar_width = .8

    # if true, shows the table of values under the plot, if false hides this table
    plot.show_table = False

def plotting_demo_diaconis_hicks_compute():
    # When does this slow down?
    # n=m=7 runs smoothly 
    # n=m=8 runs slowly
    n = 7
    m = 7
    plot = Plot()

    # car lets you change which car you are looking at. 
    # included for future plots where order might matter 
    plot.iterate_all_pref_lists(n,m, car = 1, stat = "pref")
    plot.plot()

def plotting_demo_diaconis_hicks_sample():
    # When does this slow down?
    # idk - you figure it out. If you are sampling, you might want to save too
    n = 100
    m = 100
    plot = Plot()

    # car lets you change which car you are looking at. 
    # included for future plots where order might matter 
    plot.sample_random_pref_lists(n,m, sample = 1000, car = 1, stat = "pref")
    
    # only plots categories which have 10 or more entries
    # this is important for sampling so you can look at the plots easily 
    plot.plot(plot.threshold(10))

def plotting_list_of_stats():
    """
    all of these statistics work for either random sampling or iterating through
    this is really wimpy compared to the current list of stats I have
    TODO - expand this list of implemented stats
    """
    plot = Plot()
    plot.iterate_all_pref_lists(7,7, car = 1, stat = "pref")
    plot.iterate_all_pref_lists(7,7, stat = "lucky")

def plotting_demo_Ndlambda():
    # When does this slow down?
    # n = m = 11 runs but slowlish
    # n = m = 12 runs but slower
    # n = m = 13 runs but even slower
    n = 10
    m = 10

    plot = Plot()

    # boolean value indicates weather you should multiply by the number of elements in the orbit
    plot.decompose_modules(n,m,False)
    plot.plot()

def plotting_demo_Ndlambda_sorted():
    # When does this slow down?
    # n = m = 11 runs but slowlish
    # n = m = 12 runs but slower
    # n = m = 13 runs but even slower
    n = 10
    m = 10

    plot = Plot()

    # boolean value indicates weather you should multiply by the number of elements in the orbit
    plot.decompose_modules(n,m,False)
    plot.sort_by_total()
    plot.plot()

def plotting_demo_saving_loading():
    plot = Plot()
    plot.sample_random_pref_lists(7,7, sample = 1000, car = 1, stat = "pref")
    plot.save("demo.npz")

    plot1 = Plot.load("demo.npz")

def iteractive_plot_demo():
    ip = InteractivePlot(7,7,True)
    ip.plot_mean()
    ip.slider_plot()
    #TODO show how to change statistic of interest

#################
# Cyclic Decomp #
#################

#TODO add in demos for cyclic decomposition class - useful for random walks based on convolution

def cyclic_decomp_demo():
    """
    """
    stats = IterateStats(5,5,True)
    stat = CnmStat(5,5, stats.disp_i[0])
    stat.print_by_value()           # Prints by values of fourier coefficients
    stat.print_by_value("stats")    # Prints by values of statistic
    stat.print_by_value("basis")    # Prints by values of modified basis
    stat.average_Sm()
    stat.print_by_value("stats")


