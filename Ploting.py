"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
Making Histograms From Parking Functions

TODO - list of improvements
Finish separating plotting from counting
"""

from Parking import * 

import matplotlib.pyplot as plt
import numpy as np
import os

class Ploting:

    def __init__(self, cm = plt.cm.Blues, figsize = None, bar_width = None):
        """
        An object for managing the plot creation
        """
        # varibales having to do with appearance
        self.cm = cm
        self.figsize = figsize
        self.bar_width = bar_width

        # variables having to do with data
        self.counts = None
        self.title = ""
        self.labels = None
        self.categories = None

    def plot(self):
        """
        Inputs:
            n - number of parking spots 
            m - number of cars trying to park
            stat - the thing to be plotted
        """
        Ploting.plot_counts_by_category(self.counts, self.title, self.labels, 
                                        self.categories, self.cm, self.figsize, 
                                        self.bar_width)

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
        """
        # Sets defaults
        m = m or n

        index = car or 1
        index -= 1

        stat = "pref"

        self.counts = [[0]*(n if stat == "pref" else m) for i in range(m)]

        # calculate counts
        park = Park([Car(1) for i in range(m)], n)
        for i in range(n**m):
            if stat == "pref":
                self.counts[m - park.parkability()][park.cars[index].preference-1] += 1
            elif stat == "lucky":
                self.counts[m - park.parkability()][park.lucky() - 1] += 1
            park.next()
        
        # set the rest of the relavent variables
        self.title = "n = " + str(n) + " spots , m = " + str(m) + " cars, " + ("Preference" if stat == "pref" else "Lucky")
        self.labels = None # Since labels should be integers which are the index in the array + 1
        self.categories = None # Since both of the stats use defect as their category
        
    def decompose_modules(self, n, m):
        """
        Inputs:
            n - number of parking spots 
            m - number of cars trying to park
        Outputs:
            Sets all variables relating to plotting
            This separates the orbits of $S_n$ into the corresponding defect categories
            Calculates $N_d(\lambda)$, which gives module decomposition (see 5/30)
        """
        data = Ploting.defect_partition_maps(n,m)

        partitions = list(data[0].keys())
        partitions.sort()

        self.counts = []
        for i in range(len(data)):
            L = []
            for j in range(len(partitions)):
                if partitions[j] in data[i]:
                    L.append(data[i][partitions[j]])
                else:
                    L.append(0)
            self.counts.append(L)

        self.title = "n = " + str(n) + " spots , m = " + str(m) + " cars, " + "number of orbits of $S_\lambda$"
        self.labels = [Ploting.tableau_vis(p, "m") for p in partitions]
        self.categories = None #Categories is none since categories is just defect
   
    # # # # # # # # # #
    # STATIC METHODS  #
    # # # # # # # # # #
 
    # The method that actually does all of the plotting
    @staticmethod
    def plot_counts_by_category(counts, title = None, labels = None, categories = None, cm = None, figsize = None, bar_width = None):
        """
        Inputs: Counts - 2d array. First index is category, second index is value of interest, entry is counts
                Labels - 1d array. Lables for the value of interest
                Category - 1d array. Gives names for category
        Output: Generates and shows 2 plots:
            1. A stacked histogram with a table of counts underneath
            2. A set of histgrams by defect which 
        """

        # default values and data validation
        labels = labels or [i + 1 for i in range(len(counts[0]))]
        assert len(labels) == len(counts[0])

        categories = categories or ["defect " + str(i) for i in range(len(counts))]
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
        colors = np.flip(cm(np.linspace(.5, 1, len(counts))), 0)

        # Bar chart 1 - stacked
        for d in range(len(counts)):
            plt.bar(index, counts[d], bar_width, bottom=y_offset, color=colors[d])
            y_offset = y_offset + counts[d]
        
        ax.set_xlim(-width+.5 ,len(index)+.5+width)
        
        # Table under bar chart 1
        # Note that at this point y_offset is an array of the totals
        the_table = plt.table(cellText = counts + [y_offset],
                            rowLabels  = np.append(categories, "totals"),
                            rowColours = np.vstack([colors, [0,0,0,0]]),
                            colLabels  = labels,
                            loc='bottom')
        
        the_table.scale(1, 1.2)
        cellDict = the_table.get_celld()
        for i in range(0,len(labels)):
            cellDict[(0,i)].set_height(.08)
        
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10) 
        
        plt.subplots_adjust(bottom=.07 + 0.03 * len(counts))
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
            d = m - p.parkability()
            mu = Ploting.orbit_type(pref_list)
            if mu in answer[d]:
                answer[d][mu] += 1
            else: 
                answer[d][mu] = 1
            Ploting.next_orbit_rep(pref_list, n)
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

        
