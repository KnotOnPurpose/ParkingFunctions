"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
Parking Function Variation Calculations

TODO 
Modify code to optimize:
    include a variable self.parked - tells if the full parking process has happened
    include the trick of sorting the cars and taking the max difference of index and preference
    idk how much of a speed up that is - O(n^2) worst case to O(nlog n), but
    I think expected runtime of parking is better than O(n^2)
"""

import random
import numpy as np

class Park:
    def __init__(self, cars: list, n = None, circular = False):
        """
        Park represents an single line of cars trying to park.
        Variables:

        self.n - the number of parking spots

        self.cars - a list whose contents represents the preferences of each car 
            either a list of car objects or a list of integers

        self.displacement - a list of the displacement where displacement[i] corresponds to cars[i]

        self.lot - the eventual conetents of the parking lot
        """
        self.n = n or len(cars)

        self.cars = []
        if type(cars[0]) == type(Car(1)): 
            #TODO do this correctly - 97% sure that this is a shortcut, not the right way to do this
            self.cars = cars
        elif type(cars[0]) == int or type(cars[0]) == np.int_:
            self.cars = []
            for i in range(len(cars)):
                self.cars.append(Car(cars[i], circular = circular))

        self.displacement = []
        self.park()

    def park(self):
        """
        Based on the list self.cars, updates self.lot to be the contents of the lot after all of the cars park in sequence
        """
        self.lot = [None] * self.n
        self.displacement = []
        for i in range(len(self.cars)):
            s = self.cars[i].spot(self.lot)
            if s != None:
                self.lot[s] = i + 1
            self.displacement.append(self.cars[i].displacement)

    def displacement_moment(self, i = 1):
        """
        Returns the i-th moment for the distribution of displacements
        """
        pass #TODO
    
    def displacement_total(self):
        """
        Returns the total displacement summed over all of the cars
        """
        return sum(self.displacement)

    def lucky(self):
        """
        Returns the number of lucky cars
        """
        return len(list(filter(lambda x: x == 0, self.displacement)))

    def parkability(self):
        """
        Returns the number of cars parked
        """
        return len(list(filter(lambda x: x != None, self.lot)))
    
    def defect(self):
        """
        Returns the defect associated with the preference list
        """
        return len(self.cars) - self.parkability()
    
    def apply_wr(self, L : list, s, type):
        """
        Returns a new park object corresponding to the wreath product element $type \wr S_n$ applied to the current park item 
        L : list of group elements of the type given
        s : an element of $S_n$ in one line notation represented as an array of integers
        type: either $S_n$ or $C_n$
        """

        new_cars = [None] * len(self.cars)
        for i in range(len(self.cars)):
            new_cars[s[i] - 1] = self.cars[i].apply_group(L[s[i] - 1], type)
        return Park(new_cars, self.n)
    
    def next(self):
        """
        Modifies the current object to a new object which is the next possible preference in an iterative order
        Helpful for iterating through all possible preference lists 
        NOTE: This only works for default preference type
        """
        i = 0
        while(True):
            self.cars[i] = self.cars[i].apply_group(1, "C_n", self.n)
            if self.cars[i].preference == 1:
                i += 1
                if i == len(self.cars):
                    break
            else:
                break
        self.park()

    def __str__(self):
        return str([str(c) for c in self.cars]) + " -> " + str(self.lot)
    
    @staticmethod
    def random(n, m):
        """
        Input: 
            n - number of spots
            m - number of cars
        Output: 
            Returns a random car with single spot prefereneces
        """
        return Park(np.random.randint(1,n + 1, m))


class Car:
    def __init__(self, preference, pref_type = "default", circular = False):
        """
        An object which holds the car's preferrence.
        Variables:
        
        self.preference - a variable which encodes the preference. Preferences are 1-indexed (math style, not cs style)
        
        self.pref_type - a variable which encodes the type of preference for a car. 
        Options implemented include default, n-tuple, ordered-n-tuple
        default: preference is an integer
        n-tuple: preference is an array, car will choose any spot in the array before moving on 
        ordered-n-tuple: preference is an ordered array

        circular - false for a circular parking lot
                 - true for a linear probing style parking lot
                 - TODO could be an integer for requeueing after that many cars

        self.displacement - keeps track of the number of occupied spots that a car attempted to 
                            park in at the last time spot was called

        self.unused - keeps track of elements not used in displacement so that permutations can be applied later
        """
        self.pref_type = pref_type
        self.preference = preference
        self.circular = circular
        self.displacement = 0
        if self.pref_type == "ordered-n-tuple":
            self.unedited = [preference[i] for i in range (len(preference))]

        # For the $n$-tuples, orders the elements in ascending order of spots that they would take
        if self.pref_type == "n-tuple":
            self.preference.sort()
        if self.pref_type == "ordered-n-tuple":
            assert len(preference) > 1
            current = self.preference[0]
            i = 1
            while i < len(self.preference):
                if self.preference[i] < current:
                    self.preference = self.preference[:i] + self.preference[i + 1:]
                else:
                    i += 1

    def spot(self, lot):
        """
        Given the contents of the current parking lot, returns the index in the array that the car would take
        """
        if self.pref_type == "default":
            # First checks the spots in the n-tuple, then cascades to the spots afterwards
            self.displacement = 0
            if lot[self.preference-1] == None:
                return self.preference-1
            self.displacement += 1
            for i in range(self.preference, len(lot)):
                if lot[i] == None:
                    return i
                self.displacement += 1
            if self.circular != False:
                for i in range(self.preference):
                    if lot[i] == None:
                        return i
                    self.displacement += 1
        if self.pref_type == "n-tuple" or self.pref_type == "ordered-n-tuple":
            # First checks the spots in the n-tuple, then cascades to the spots afterwards
            self.displacement = 0
            for i in range(len(self.preference)):
                if lot[self.preference[i]-1] == None:
                    return self.preference[i]-1
                self.displacement += 1
            for i in range(self.preference[-1], len(lot)):
                if lot[i] == None:
                    return i
                self.displacement += 1

    def apply_group(self, g, type, n = None):
        """
        Returns a car which is the result of acting on the object with the group element g
        Inputs:
            g - the group element. 
            For elements of S_n, this should be an array of integers representing one line notation
            For elements of C_n, this should be an integer
            
            type - what group the element comes from (either S_n or C_n)
        """
        if self.pref_type == "default":
            if type == "S_n":
                new_pref = g[self.preference - 1] 
            if type == "C_n":
                new_pref = (self.preference + g)%n
                if new_pref == 0:
                    new_pref = n
        
            return Car(new_pref, self.pref_type, self.circular)

        if self.pref_type == "n-tuple" or self.pref_type == "ordered-n-tuple":
            if self.pref_type == "ordered-n-tuple":
                new_pref = self.unedited
            elif self.pref_type == "n-tuple":
                new_pref = self.preference

            for i in range(len(new_pref)):
                if type == "S_n":
                    new_pref[i] = g[new_pref[i] - 1] 
                if type == "C_n":
                    new_pref[i] = (new_pref[i] + g)%n 
                    if new_pref[i] == 0:
                        new_pref[i] == n
            return Car(new_pref, self.pref_type)

    def __str__(self) -> str:
        return str(self.preference)

