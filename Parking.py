"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
Parking Function Variation Calculations

TODO clean up documentation

TODO
code structure wise, it would be good to pull the statistics calculating stuff from plotting
and just use the calculate statistics object for all of that.
"""

import random
import numpy as np

class IterateStats:
    """
    This is an object which calculates a lot of useful statistics all at once
    TODO add an option to sample/record counts which is more geared towards plotting
    Because it would be good for all of the calculating statistics to happen in the same place

    TODO finish writing out documentation descriptions
    variables:
    settings
    self.n
    self.m
    self.circular - boolean
    self.sample - either None or a number

    calculated stuff

    Graded kind of stats
    self.disp_i - row is displacement, col is pref list. ex lucky is row 0
    self.wants_k - 
    self.lel_i - 

    Single Stats
    self.max_disp - 
    self.total_disp - 
    self.repeats -
    self.defect -
    self.longest_prime -
    self.last_start_point - 

    methods:
    calculate_all() - for when you really just want to calculate all the things
                    - also a good place to look for all of the things

    get_graded(stat, graded_by) - useful for plotting 
                                - returns array of counts AND 
    """
    def __init__(self, n, m = None, circular = True):
        """
        Upon creating this object, many useful statistics are calculated
        """
        self.n = n
        self.m = m or n
        self.circular = circular

        # Graded stats
        self.disp_i = np.zeros((self.m, self.n**self.m), int)
        self.wants_k = np.zeros((self.n, self.n**self.m), int)
        self.lel_i = np.zeros((self.m, self.n**self.m), int)
        self.passed_i = np.zeros((self.m, self.n**self.m), int)

        # single stats
        self.max_disp = np.zeros(self.n**self.m, int)
        self.total_disp = np.zeros(self.n**self.m, int)
        self.repeats = np.zeros(self.n**self.m, int)
        self.defect = np.zeros(self.n**self.m, int) 
        self.longest_prime = np.zeros(self.n**self.m, int)
        self.last_start_point = np.zeros(self.n**self.m, int)

        #TODO could add more stats derived from resulting permutation

    def iterate(self):
        """
        Calculates displacement_counts and total_disp
        """
        park = Park([1] * self.m, self.n, circular = self.circular)

        for i in range(self.n**self.m):
            t = 0
            for j in range(self.m):
                self.disp_i[j][i] = park.displacement.count(j)
                self.wants_k[j][i] = park.cars.count(j)
                self.lel_i[j][i] = park.cars.count(park.cars[j])
                self.passed_i[j][i] = park.passed.count(j)
                t += self.disp_i[j][i]
                if t == self.m and self.max_disp[i] == 0:
                    self.max_disp[i] = j
            for j in range(self.m,self.n):
                self.wants_k[j][i] = park.cars.count(j)
            self.defect[i] = park.defect()
            self.repeats[i] = park.repeats()
            self.longest_prime[i] = max(park.prime_lengths())
            self.last_start_point[i] = max(park.start_points)
            park.next()

        self.total_disp = np.transpose(np.matrix.transpose(self.disp_i).dot(np.arange(self.m))) 

    def get_graded():
        """
        TODO - would be useful for plots
        """   
    
    """ Tested a different way of iterating. it was slower.
    def iterate2(self):
        self.total_disp = np.zeros([self.n]*self.m)
        self.displacement_counts = np.zeros((self.m, self.n**self.m), int)

        i = 0
        for idx, x in np.ndenumerate(self.total_disp):
            park = Park(idx, zero_index=True)
            for j in range(self.m):
                self.displacement_counts[j][i] = park.displacement.count(j)
            i += 1
    """

class Park:
    """
    Variables:

    cars - a list of Car objects capturing the preference of each car
    
    lot - the result of the parking process
    
    displacement - the displacement of each car during the parking process
    
    passed - the nunmber of cars which attempted to park in the given spot
    
    break_points - the points in the list which are break points (occupied by a car, but no other car tries to park there)
    
    start_points - the start point corresponding to a given break point. 
                    Note that if the preference type is not circular, the number of start points could be 1 larger than the number of break points
    """
    def __init__(self, cars: list, n = None, circular = False, zero_index = False):
        """ 
        Park represents an single line of cars trying to park.
        Inputs:
        cars
        n
        circular
        zero_index - if this is set to true and an array of integers is provided for the cars, the array is treated as 0 indexed

        Variables:

        self.n - the number of parking spots

        self.cars - a list whose contents represents the preferences of each car 
            either a list of car objects or a list of integers

        self.displacement - a list of the displacement where displacement[i] corresponds to cars[i]

        self.lot - the eventual conetents of the parking lot
        """
        self.n = n or len(cars)
        self.circular = circular

        self.cars = []
        if type(cars[0]) == type(Car(1)): 
            self.cars = cars
        elif type(cars[0]) == int or type(cars[0]) == np.int_:
            self.cars = []
            for i in range(len(cars)):
                if zero_index:
                    self.cars.append(Car(cars[i] + 1, circular = circular))
                else:
                    self.cars.append(Car(cars[i], circular = circular))
        assert self.cars != []

        self.displacement = []
        self.park()

    def park(self):
        """
        Based on the list self.cars, updates self.lot to be the contents of the lot after all of the cars park in sequence
        """
        empty = [None] * self.n

        self.lot = [None] * self.n
        self.passed = [0] * self.n
        self.displacement = []
        for i in range(len(self.cars)):
            s = self.cars[i].spot(self.lot)
            if s != None:
                self.lot[s] = i + 1
            self.displacement.append(self.cars[i].displacement)
            for j in range(self.cars[i].spot(empty), s if s != None else self.n):
                self.passed[j%self.n] += 1
        
        # break points are where the parking lot is not empty and passed is 0
        self.break_points = np.array([i+1 for i  in range(self.n) if self.passed[i] == 0 and self.lot[i] != None], int)
        # start points are the first occupied spot after previous break point
        if not self.circular and self.lot[-1] != None and self.passed[-1] != 0:
            self.start_points = np.roll(np.append(self.break_points, self.n), 1)
        else:
            self.start_points = np.roll(self.break_points, 1)
        
        if self.circular:
            if self.start_points[0] == self.n:
                self.start_points[0] = 0
        else:
            self.start_points[0] = 0
        for i in range(len(self.start_points)):
            while self.lot[self.start_points[i]] == None:
                self.start_points[i] += 1
                if self.circular:
                    self.start_points[i] = self.start_points[i] % len(self.lot)
        self.start_points = self.start_points + 1 # to one index for math reasons
        

    ##################################
    # METHODS FOR GETTING STATISTICS #
    ##################################
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

    def disp_i(self, i):
        """
        Returns the number of cars displaced exactly i
        """
        return self.displacement.count(i)

    def lucky(self):
        """
        Returns the number of lucky cars
        """
        return self.disp_i(0)
    
    def defect(self):
        """
        Returns the defect associated with the preference list
        """
        #assert self.passed[-1] == len(self.cars) - len(list(filter(lambda x: x != None, self.lot)))
        if self.circular:
            return 0
        else:
            return self.passed[-1]
    
    def prime_lengths(self):
        """
        Returns the lengths of the prime segments composing the preference list
        Note that the sum of the components will always be $m$ (the number of cars)
        """
        if len(self.break_points) == len(self.start_points):
            return (self.break_points - self.start_points)%self.n + 1
        else:
            arr = (self.break_points - self.start_points[:-1]) + 1
            return np.append(arr, len(self.cars) - sum(arr))

    def repeats(self):
        """
        Returns the number of repeats in the preference list
        """
        cnt = 0
        for i in range(len(self.cars) - 1):
            if self.cars[i].preference == self.cars[i+1].preference:
                cnt += 1
        return cnt


    #########################################
    # METHODS FOR MODIFYING PREFERENCE LIST #
    #########################################
    def apply_wr(self, L : list, s = None, type = "C_n"):
        """
        Returns a new park object corresponding to the wreath product element $type \wr S_n$ applied to the current park item 
        L : list of group elements of the type given
        s : an element of $S_n$ in one line notation represented as an array of integers
        type: either $S_n$ or $C_n$
        """
        if s == None:
            s = list(range(len(self.cars)))

        new_cars = [None] * len(self.cars)
        for i in range(len(self.cars)):
            new_cars[s[i] - 1] = self.cars[i].apply_group(L[s[i] - 1], type, self.n)
        self.cars = new_cars
        self.park()
    
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

    def walk(self, steps):
        """
        Does a random walk with the number of steps indicated
        Input: 
            steps - the number of steps to take
            TODO - modify the arguments to match conversation with Prof O:
                    Would be nice to be able to give a distribution for each parking spot, 
                    resulting distribution would be the tensor of all of them
                    Could also be interesting to have a couple of options
                      that don't necessarily play nice with fourier transform, but take nice discrete steps
        Result:
            modifies the object to be a new object after taking steps steps.
        """
        indices = range(len(self.cars))
        for i in range(steps):
            ind = np.random.choice(indices)
            self.cars[ind] = self.cars[ind].apply_group(np.random.choice([-1,1]), "C_n", self.n)
        self.park()

    def __str__(self):
        return str([str(c) for c in self.cars]) + " -> " + str(self.lot)
    
    @staticmethod
    def random(n, m, circular = False):
        """
        Input: 
            n - number of spots
            m - number of cars
        Output: 
            Returns a random preference list with single spot prefereneces
        """
        return Park(np.random.randint(1,n + 1, m), n, circular)

    @staticmethod
    def random_pf(n,m):
        """
        Input: 
            n - number of spots
            m - number of cars
        Output: 
            Returns a random parking function with single spot prefereneces
        """
        intermediate = Park.random(n+1,m, circular = True)
        empty = [i for i in range(n + 1) if intermediate.lot[i] == None]
        end = np.random.choice(empty)
        return Park(np.array([(car.preference-end-1)%(n+1) for car in intermediate.cars], int))
    
    @staticmethod
    def random_ppf(n):
        """
        Input: 
            n - number of spots
            m - number of cars
        Output: 
            Returns a random prime parking function with single spot prefereneces
        """
        intermediate = Park.random(n-1,n, circular = False)
        for i in range(n):
            if intermediate.defect() == 1 and len(intermediate.prime_lengths()) == 1:
                break
            intermediate.apply_wr([1]*n)
        return Park(intermediate.cars, n, False)

    @staticmethod
    def random_def(d):
        """
        Input: 
            n - number of spots
            m - number of cars
        Output: 
            Returns a random defect d preference list with single spot prefereneces
            #TODO Actually ready to write this method now
        """
        if d == 0:
            return Park.random_pf()
        pass #TODO

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
