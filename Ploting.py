"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
Making Histograms From Parking Functions

TODO - improvements for code
separate the data manipulation from the plotting

plotting should take in a list of labels and list of things ordered by defect
- resulting graph - 2 types (same as orbits)

File management system
"""

from Parking import * 

import matplotlib.pyplot as plt
import numpy as np
import os

class Ploting:
    def __init__(self, cm = plt.cm.Blues):
        """
        An object for managing the plot creation
        """
        self.cm = cm

    ############################################################
    #   THIS SECTION IS FOR FUNCTIONS WHICH MAKE PLOTS/FILES   #
    ############################################################

    def histogram_complete(self, n, m = None, car = None, stat = "pref"):
        """
        Inspiration from Diaconis, Hicks 2017 paper
        Inputs:
            n - number of parking spots 
            m - number of cars trying to park
            index - the index of the preference to record - 
                    should not matter for counts of preference
            type - indicates the type of histogram that should be made
                choices: "pref" - makes a histogram of the preference of the first car
                         "lucky" - makes a histogram based on the number of lucky cars
        Outputs: 
            Displays a plot for the distribution of the preference of the ith car based on defect
        """
        index = car or 1
        index -= 1

        m = m or n

        # count is 2d array
        # first index represents the defect - m possibilities
        # second index represents the preference of the car - n possibilities
        #                        or the number of lucky cars - m possibilities
        counts = [[0]*(n if stat == "pref" else m) for i in range(m)]
        hist_data = [[] for i in range(m)]

        # calculate counts
        park = Park([Car(1) for i in range(m)], n)
        for i in range(n**m):
            if stat == "pref":
                counts[m - park.parkability()][park.cars[index].preference-1] += 1
                hist_data[m - park.parkability()].append(park.cars[index].preference)
            elif stat == "lucky":
                counts[m - park.parkability()][park.lucky() - 1] += 1
                hist_data[m - park.parkability()].append(park.lucky())
            park.next()
        
        print(counts)

        # plot counts - seperate plots
        bins =[.5 + i for i in range(len(counts[0]) + 1)]
        colors = [self.cm((len(hist_data)*1.5 - i)/(len(hist_data)*1.5)) for i in range(len(hist_data))]

        fig, axs = plt.subplots(len(counts), 1, sharex=True, tight_layout=True, figsize = (5, 6))
        fig.suptitle( "n = " + str(n) + " spots , m = " + str(m) + " cars, " + ("Preference" if stat == "pref" else "Lucky"))
        for i in range(len(counts)):
            num, bins, patches = axs[len(axs) - 1 - i].hist(hist_data[i], bins = bins, color = colors[i])
            axs[len(axs) - 1 - i].set_title("defect " + str(i))
        
        # plot counts - same plot
        fig, ax = plt.subplots(figsize =(5, 6))
        fig.suptitle( "n = " + str(n) + " spots , m = " + str(m) + " cars, " + ("Preference" if stat == "pref" else "Lucky"))
        ax.hist(hist_data, bins = bins, color = colors, density=True, histtype='bar', stacked=True)
        
        plt.show()

    def num_defect_partition(self, n, m):
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
            Ploting.increment_orbit_representative(pref_list)
            if pref_list == [1] * m:
                break
        return answer

    def plot_num_defect_partition(self, n, m):
        """
        Inputs:
            n - number of parking spots 
            m - number of cars trying to park
        Outputs:
            Plots!
        """
        data = self.num_defect_partition(n,m)

        partitions = list(data[0].keys())
        partitions.sort()

        partition_counts = []
        for i in range(len(data)):
            L = []
            for j in range(len(partitions)):
                if partitions[j] in data[i]:
                    L.append(data[i][partitions[j]])
                else:
                    L.append(0)
            partition_counts.append(L)

        fig, ax = plt.subplots(figsize =(5, 6))
        fig.suptitle( "n = " + str(n) + " spots , m = " + str(m) + " cars, " + "number of orbits of $S_\lambda$")

        index = np.arange(len(partitions)) + 1
        y_offset = np.zeros(len(partitions))
        bar_width = .8  
        width = 0
        colors = np.flip(self.cm(np.linspace(.5, 1, len(partition_counts))), 0)

        for d in range(len(partition_counts)):
            plt.bar(index, partition_counts[d], bar_width, bottom=y_offset, color=colors[d])
            y_offset = y_offset + partition_counts[d]
        
        #Table
        ax.set_xlim(-width+.5 ,len(index)+.5+width)
            
        totals = np.zeros(len(partition_counts[0]), dtype = int)
        for row in partition_counts:
            totals += row

        the_table = plt.table(cellText=partition_counts + [totals],
                            rowLabels=np.append(np.arange(len(partition_counts)), "totals"),
                            rowColours=np.vstack([colors, [0,0,0,0]]),
                            colLabels=[Ploting.tableau_vis(p, "m") for p in partitions],
                            loc='bottom')
        
        the_table.scale(1, 1.2)
        cellDict = the_table.get_celld()
        for i in range(0,len(partitions)):
            cellDict[(0,i)].set_height(.1)
        
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10) 
        
        plt.subplots_adjust(bottom=.07 + 0.03 * len(partition_counts))
        plt.xticks([])

        # SECOND FIGURE - visualizations split
        plt.figure(1)
        fig.suptitle( "n = " + str(n) + " spots , m = " + str(m) + " cars, " + "number of orbits of $S_\lambda$")
        
        fig, axs = plt.subplots(len(partition_counts), 1, sharex=True, tight_layout=True, figsize = (5, 6))
        for i in range(len(partition_counts)):
            axs[len(axs) - 1 - i].bar(index, partition_counts[i], 
                                        color = colors[i])
            axs[len(axs) - 1 - i].set_title("defect " + str(i))
        plt.xticks(index, labels = [Ploting.tableau_vis(p, "m") for p in partitions])
        
        plt.show()


    # # # # # # # # # #
    # STATIC METHODS  #
    # # # # # # # # # #

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
    def next_orbit_rep(arr):
        """
        These lists are a set of representatives for the orbits of $S_n$ in preference lists
        input: an array which represents a preference list (ascending order, orbit representative). 
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

        
