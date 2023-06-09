"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
Making Histograms From Parking Functions

TODO - list of improvements
- develop a system for generating many plots at once (save optional)
- update the plot_counts_by_category so that if there are a Lot of categories,
    the plot is given in multiple columns. (maybe just have columns as an input...)
- add in a function which can take in a CnmStat and 
"""

from Parking import * 
from CyclicDecomp import *

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import os

class Plot:

    # I only ever need small values of this... so I borrowed the list from OEIS A000041
    NUM_PARTITIONS = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575, 1958, 2436, 3010, 3718, 4565, 5604]

    def __init__(self, cm = plt.cm.Blues, figsize = None, bar_width = None, show_table = True):
        """
        An object for managing the plot creation
        Inputs: cm - default is Blues
                figsize - default is (5,6)
                bar_width - default is 1
                show_table - default is True
        """

        # varibales having to do with appearance
        self.cm = cm
        self.figsize = figsize
        self.bar_width = bar_width
        self.show_table = show_table

        # variables having to do with data
        self.counts = None
        self.title = ""
        self.labels = None
        self.categories = None

    def plot(self, rows = None):
        """
        Inputs:
            rows - which rows to plot. Particularly helpful for sampling
        """
        if type(rows) == type(None):
            Plot.plot_counts_by_category(self.counts, self.title, self.labels, 
                                        self.categories, self.cm, self.figsize, 
                                        self.bar_width, self.show_table)
        else: 
            Plot.plot_counts_by_category(self.counts[rows], self.title, self.labels, 
                                        self.categories[rows], self.cm, self.figsize, 
                                        self.bar_width, self.show_table)
        
    def save(self, file_name):
        """
        Saves the important data to the "saved" folder with the given file name
        Inputs: file_name - the name of the file within the saved folder
        """
        np.savez("saved/" + file_name, counts = self.counts,
                 labels = self.labels,
                 categories = self.categories,
                 other_data = np.array([self.title, self.cm, self.figsize, 
                                        self.bar_width, self.show_table])
                 )

    @staticmethod
    def load(file_name):
        """
        Loads the data from a npz file in the saved folder
        Inputs: file_name - the name of the file to be loaded
        Outputs: an object with the loaded plot parameters
        """
        plot = Plot()

        load_data = np.load("saved/" + file_name, allow_pickle=True)

        plot.counts = load_data["counts"]
        plot.labels = load_data["labels"]
        plot.categories = load_data["categories"]
        other_data = load_data["other_data"]

        plot.title = other_data[0]
        plot.cm = other_data[1]
        plot.figsize = other_data[2]
        plot.bar_width = other_data[3]
        plot.show_table = other_data[4]

        return plot

    ########################################################################
    #   THIS SECTION IS FOR FUNCTIONS WHICH SET VARIBLES RELATED TO PLOT   #
    ########################################################################
        
    def iterate_all_pref_lists(self, n, m = None, car = None, stat = None):
        """
        Inspiration from Diaconis, Hicks 2017 paper
        Inputs:
            n - number of parking spots 
            m - number of cars trying to park
            car - the index of the preference to record - 
                    should not matter for counts of preference
            stat - indicates the type of histogram that should be made
                choices: "pref" - makes a histogram of the preference of the first car
                         "lucky" - makes a histogram based on the number of lucky cars
        Result: 
            Sets all variables related to plotting
        #TODO maybe make this lean on IterateStats more
        """
        # Sets defaults
        m = m or n

        index = car or 1
        index -= 1

        stat = stat or "pref"
        assert stat in ["pref", "lucky"]

        self.counts = [[0]*(n if stat == "pref" else m) for i in range(m)]

        # calculate counts
        park = Park([Car(1) for i in range(m)], n)
        for i in range(n**m):
            if stat == "pref":
                self.counts[park.defect()][park.cars[index].preference-1] += 1
            elif stat == "lucky":
                self.counts[park.defect()][park.lucky() - 1] += 1
            park.next()
        
        # set the rest of the relavent variables
        self.title = "n = " + str(n) + " spots , m = " + str(m) + " cars, " + ("Preference" if stat == "pref" else "Lucky")
        self.labels = None # Since labels should be integers which are the index in the array + 1
        self.categories = None # Since both of the stats use defect as their category
        
    def sample_random_pref_lists(self, n, m = None, sample = None, car = None, stat = None):
        """
        Samples from preference lists uniformly at random and evaluates the given statistic
        TODO maybe make this lean on a different class for random sampling? not high priority
        """
        # Sets defaults
        m = m or n

        sample = sample or 1000

        index = car or 1
        index -= 1

        stat = "pref"

        self.counts = np.zeros((m, n if stat == "pref" else m), int)

        # calculate counts
        park = None
        for i in range(sample):
            park = Park.random(n, m)
            if stat == "pref":
                self.counts[park.defect()][park.cars[index].preference-1] += 1
            elif stat == "lucky":
                self.counts[park.defect()][park.lucky() - 1] += 1
        
        # set the rest of the relavent variables
        self.title = "Sampling = " + str(sample) 
        self.title += ", n = " + str(n) + " spots , m = " + str(m) 
        self.title += " cars, " + ("Preference" if stat == "pref" else "Lucky")
        self.labels = None # Since labels should be integers which are the index in the array + 1
        self.categories = np.array(["defect " + str(i) for i in range(len(self.counts))])
                
    def decompose_modules(self, n, m, size_orbit = False):
        """
        Inputs:
            n - number of parking spots 
            m - number of cars trying to park
            size_orbit - if this is true, multiplies by the size of the orbit
        Outputs:
            Sets all variables relating to plotting:
               - data is the number of partitions 
               - categories are defect
               - labels are partitions 
            This separates the orbits of $S_n$ into the corresponding defect categories
            Calculates $N_d(\lambda)$, which gives module decomposition (see 5/30)
        """
        data = Plot.defect_partition_maps(n,m)

        partitions = list(data[0].keys())
        partitions.sort()

        self.counts = np.zeros((len(data), len(partitions)), dtype=int)
        for i in range(len(data)):
            for j in range(len(partitions)):
                if partitions[j] in data[i]:
                    self.counts[i][j] = data[i][partitions[j]]
                    
        size = np.array([Plot.size_orbit(partitions[i]) for i in range(len(partitions))])
        if size_orbit:
            self.counts = np.multiply(self.counts, size)

        self.title = "n = " + str(n) + " spots , m = " + str(m) + " cars, " 
        self.title += ("total sizes" if size_orbit else "number") +  " of orbits of $S_\lambda$"
        self.labels = [Plot.tableau_vis(p, "m") for p in partitions]
        self.categories = None #Categories is none since categories is just defect
   
    ############################################
    # HELPER METHODS FOR POST DATA FILTER/SORT #
    ############################################
    def sort_by_total(self):
        """
        Reorders the columns by their total - useful for looking at decomplse_module outputs 
        """
        
        new_counts = np.transpose(self.counts)
        totals = [np.sum(new_counts[i]) for i in range(len(new_counts))]
        order = np.argsort(totals, kind = "stable")

        new_counts = new_counts[order]

        self.counts = np.transpose(new_counts)
        self.labels = np.array(self.labels)[order]
        
    def threshold(self, threshold):
        """
        returns an array which is true if the value is over the threshold
        this is 
        input: 
            threshold - the cutoff value
        output:
            a list which is true if the sum of the counts is > threshold
        """
        rows = [True]*len(self.counts)
        for i in range(len(self.counts)):
            if np.sum(self.counts[i]) <= threshold:
                rows[i] = False

        return rows

    # # # # # # # # # #
    # STATIC METHODS  #
    # # # # # # # # # #
 
    # The method that actually does all of the plotting
    @staticmethod
    def plot_counts_by_category(counts, title = None, labels = None, categories = None, 
                                cm = None, figsize = None, bar_width = None, show_table = True):
        """
        Inputs: Counts - 2d array. First index is category, second index is value of interest, entry is counts
                Title - string. The title for the plot. Default value ""
                Labels - 1d array. Lables for the value of interest. Default value - integers (1 indexed)
                Category - 1d array. Gives names for category. Default value - [defect 0, defect 1, ...]
                cm - default is Blues
                figsize - default is (5,6)
                bar_width - default is 1
                show_table - default is True
        Output: Generates and shows 2 plots:
            1. A stacked histogram with a table of counts underneath if show_table is true
            2. A set of histgrams by category
        """

        # default values and data validation
        if type(labels) == type(None):
            labels = [i + 1 for i in range(len(counts[0]))]
        assert len(labels) == len(counts[0])

        if type(categories) == type(None):
            categories = ["defect " + str(i) for i in range(len(counts))]
        assert len(categories) == len(counts)

        title = title or ""
        cm = cm or plt.cm.Blues
        figsize = figsize or (5,6)
        bar_width = bar_width or 1

        # Figure set up
        fig, ax = plt.subplots(figsize = figsize)
        fig.suptitle(title)

        # Bar chart variables set up 
        index = np.arange(len(labels)) + 1
        y_offset = np.zeros(len(labels), dtype = int)
        width = 0
        colors = np.flip(cm(np.linspace(.3, 1, len(counts))), 0)

        # Bar chart 1 - stacked
        for d in range(len(counts)):
            plt.bar(index, counts[d], bar_width, bottom=y_offset, color=colors[d])
            y_offset = y_offset + counts[d]
        
        ax.set_xlim(-width+.5 ,len(index)+.5+width)
        
        # Table under bar chart 1
        # Note that at this point y_offset is an array of the totals
        if show_table:
            the_table = plt.table(cellText = np.vstack([counts, y_offset]),
                                rowLabels  = np.append(categories, "totals"),
                                rowColours = np.vstack([colors, [0,0,0,0]]),
                                colLabels  = labels,
                                loc='bottom')
            
            the_table.scale(1, 1.2)
            cellDict = the_table.get_celld()
            for i in range(0,len(labels)):
                cellDict[(0,i)].set_height(.08)
            
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(12) 
            
            plt.subplots_adjust(bottom=.12 + 0.02 * (len(counts) + 1))
            plt.xticks([])

        # SECOND FIGURE - visualizations split
        plt.figure(1)
        fig, axs = plt.subplots(len(counts), 1, sharex=True, tight_layout=True, figsize = figsize)
        fig.suptitle(title)
        
        for i in range(len(counts)):
            axs[len(axs) - 1 - i].bar(index,counts[i], bar_width,
                                        color = colors[i])
            axs[len(axs) - 1 - i].set_title(categories[i])
        plt.xticks(index, labels = labels)
        
        plt.show()

    @staticmethod
    def defect_partition_maps(n, m):
        """
        Based on calculations 5/24/2023
        Inputs:
            n - number of parking spots 
            m - number of cars trying to park
        Outputs: 
            an array of maps from partitions to counts
            index of the array indicates the defect number
            key is a partition - tuple in descending order

            The counts are the number of orbits of S_n with that particular partition type
            ie $N_d(\lambda)$
        """
        m = m or n 

        pref_list = [1] * m 

        answer = [{} for i in range(m)]
        while True:
            p = Park(pref_list, n)
            d = p.defect()
            mu = Plot.orbit_type(pref_list)
            if mu in answer[d]:
                answer[d][mu] += 1
            else: 
                answer[d][mu] = 1
            Plot.next_orbit_rep(pref_list, n)
            if pref_list == [1] * m:
                break
        return answer

    @staticmethod
    def tableau_vis(partition, format):
        """
        Given a partition returns a visualization or shortened version of the partition
        Inputs:
        partition - the partition to format
        format - either "*" to format the partition as a tableau of astrix
                    or  "m" to format the partition with math shorthand
        """
        if format == "*":
            s = ""
            for p in partition:
                s += "*"*p + "\n"
            s = s[:-1]
            return s

        if format == "m":
            s = ""

            counts = {}
            for i in range(len(partition)):
                if partition[i] in counts:
                    counts[partition[i]] += 1
                else:
                    counts[partition[i]] = 1
            
            L = list(counts.keys())
            L.sort()

            for i in range(len(L)):
                s += str(L[i])
                if counts[L[i]] != 1:
                    s += "^" + str(counts[L[i]]) + " "
            return "$" + s + "$"
                
    @staticmethod 
    def next_orbit_rep(arr, n):
        """
        These lists are a set of representatives for the orbits of $S_n$ in preference lists
        input: 
            arr - a preference list in ascending order which is an orbit representative.
            n   - the number of parking spaces 
        result: modifies the array to be the next orbit representative.
        """
        # j is the first index which is not n
        j = 0
        while arr[j] == n:
            j += 1
            # if the entire list is n, the incremented list should wrap back around to the start
            if j == len(arr):
                for i in range(len(arr)):
                    arr[i] = 1
                return 
        
        arr[j] += 1
        for i in range(j):
            arr[i] = arr[j]
        return

    @staticmethod
    def orbit_type(arr):
        """
        input: arr which is a preference list
        output: a partition which describes the number of each element in the preference list
        """
        counts = {}
        for i in range(len(arr)):
            if arr[i] in counts:
                counts[arr[i]] += 1
            else:
                counts[arr[i]] = 1
        
        partition = list(counts.values())
        partition.sort()
        partition.reverse()
        return tuple(partition)

    @staticmethod
    def size_orbit(arr):
        """
        Returns the size of the orbit of $S_m$ of a preference list with multiplicity partition corresponding to arr
        Input: arr - gives a partition of $m$
        Output: Size of the orbit: $m!/\prod_{i = 1}(\lambda_i!)$
        """
        ans = np.math.factorial(np.sum(arr))
        for part in arr:
            ans = ans/np.math.factorial(part)
        return int(ans)

class InteractivePlot:

    def __init__(self, n, m, circular = None):
        self.n = n
        self.m = m or n
        self.circular = circular
    
        if circular == None:
            self.circular = True

        self.stats = IterateStats(self.n, self.m, self.circular)

        dat = np.zeros(self.n**self.m)
        dat[0] = 1
        self.start = CnmStat(self.n,self.m, dat, arr_type="stat")
        
        self.walk = CnmStat.cos(self.n,self.m) 
        self.walk.distribution()

        self.s = CnmStat(self.n,self.m, self.stats.disp_i[0]) #lucky

    def slider_plot(self):
        # https://matplotlib.org/stable/gallery/widgets/slider_demo.html
        def f(x, steps):
            dwt = CnmStat(self.n,self.m)
            dwt.set_fourier(self.start.fourier * (self.walk.fourier**int(steps)))
        
            ans = np.zeros(len(x))
            for i in range(len(ans)):
                ans[i] = sum(dwt.stat[self.s.stat == x[i]])
            return ans

        x = np.arange(0, self.n + 1, 1, int)

        # Define initial parameters
        init_steps = 0

        # Create the figure and the line that we will manipulate
        fig, ax = plt.subplots()
        line, = ax.plot(x, f(x, init_steps), lw = 1)
        ax.set_xlabel('Statistic')

        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0.25, bottom=0.25)

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
        steps_slider.on_changed(update)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')

        def reset(event):
            steps_slider.reset()
        button.on_clicked(reset)

        plt.show()
        
    def plot_mean(self, max_steps):
        dwt = CnmStat(self.n,self.m)
        dwt.set_fourier(self.start.fourier)
        data = [self.start.expected_val(self.s)]
        for i in range(max_steps):
            dwt = dwt.convolve(self.walk)
            print(np.sum(dwt.stat))
            data.append(dwt.expected_val(self.s))
        
        plt.plot(data)
        plt.show()