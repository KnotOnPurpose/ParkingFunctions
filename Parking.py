"""
Author: Jasper Bown 
Email: abown@hmc.edu
Summer 2023
Parking Function Variation Calculations

TODO
code structure wise, it would be good to pull the statistics calculating stuff from plotting
and just use the calculate statistics object for all of that.
"""

import random
import numpy as np
import os.path

class IterateStats:
    """
    This is an object which calculates a lot of useful statistics all at once
    settings
        \\
        self.n - the number of spots\\
        self.m - the number of cars\\
        self.circular - boolean to indicate one way parking or circular linear probing\\
        self.sample - either None or a number
    ___________________________________________
    calculated stuff:

    Graded kind of stats \\
    self.disp_i - number of cars displaced by row. ex lucky is row 0 \\
    self.wants_k - numper of preferences matching row. Ex ones is row 0 \\
    self.lel_i - number of spots passed by row cars. ex lucky is row 0 
    
    Single Stats \\
    self.max_disp - max displacement \\
    self.total_disp - total displacement/area \\
    self.repeats - number of preferences the same as the previous\\
    self.defect - number of cars which cannot park \\        
    self.longest_prime - length of the longest prime segment \\        
    self.last_start_point - the location of the last prime segment
    """
    
    """
    TODO 
    - add an option to sample/record counts which is more geared towards plotting
        because it would be good for all of the calculating statistics to happen in the same place

    TODO 
    could add more stats derived from resulting permutation or other stats

    """
    save_dir = "saved/IterateStats/"

    def __init__(self, n, m = None, circular = True):
        """
        Upon creating this object, many useful statistics are calculated
        """
        self.n = n
        self.m = m or n
        self.circular = circular

        # Graded stats
        self.disp_i = np.zeros((self.m, self.n**self.m), int)
        self.lel_i = np.zeros((self.m, self.n**self.m), int)
        self.passed_i = np.zeros((self.m, self.n**self.m), int)
        self.wants_k = np.zeros((self.n, self.n**self.m), int)
        self.passed_k = np.zeros((self.n, self.n**self.m), int)

        # single stats
        self.max_disp = np.zeros(self.n**self.m, int)
        self.total_disp = np.zeros(self.n**self.m, int)
        self.repeats = np.zeros(self.n**self.m, int)
        self.defect = np.zeros(self.n**self.m, int) 
        self.longest_prime = np.zeros(self.n**self.m, int)
        self.last_start_point = np.zeros(self.n**self.m, int)

        if os.path.exists(self.path()):
            print(". . . loading from file . . .")
            self.load()
        else:
            print(". . . iterating . . .")
            self.iterate()
            self.save()

    ##############################
    # Methods relating to saving #
    ##############################

    def save(self):
        """
        Saves the important data to the "saved" folder with the given file name
        Inputs: file_name - the name of the file within the saved folder
        """
        print(". . . saving . . .")
        np.savez(self.path(), 
                 disp_i = self.disp_i,
                 lel_i = self.lel_i,
                 passed_i = self.passed_i,
                 wants_k = self.wants_k,
                 passed_k = self.passed_k,
                 max_disp = self.max_disp,
                 total_disp = self.total_disp,
                 repeats = self.repeats,
                 defect = self.defect,
                 longest_prime = self.longest_prime,
                 last_start_point = self.last_start_point,
                 other_data = np.array([self.n, self.m, self.circular])
                 )
        
    def load(self):
        """
        Loads the data from a npz file in the saved folder
        Inputs: file_name - the name of the file to be loaded
        Outputs: an object with the loaded plot parameters
        """
        load_data = np.load(self.path(), allow_pickle=True)

        try:
            self.disp_i = load_data["disp_i"]
        except:
            print("disp_i not loaded")
        try:
            self.lel_i = load_data["lel_i"]
        except:
            print("lel_i not loaded")
        try:
            self.passed_i = load_data["passed_i"]
        except:
            print("passed_i not loaded")
        try:
            self.wants_k = load_data["wants_k"]
        except:
            print("wants_k not loaded")
        try:
            self.passed_k = load_data["passed_k"]
        except:
            print("passed_k not loaded")
        try:
            self.max_disp = load_data["max_disp"]
        except:
            print("max_disp not loaded")
        try:
            self.total_disp = load_data["total_disp"]
        except:
            print("total_disp not loaded")
        try:
            self.repeats = load_data["repeats"]
        except:
            print("repeats not loaded")
        try:
            self.defect = load_data["defect"]
        except:
            print("defect not loaded")
        try:
            self.longest_prime = load_data["longest_prime"]
        except:
            print("longest_prime not loaded")
        try:
            self.last_start_point = load_data["last_start_point"]
        except:
            print("last_start_point not loaded")

    def path(self):
        """
        returns the path name for the object to save/load
        """
        return self.save_dir + "n" + str(self.n) + "m" + str(self.m) + ("c" if self.circular else "") + ".npz"

    def iterate(self):
        """
        Calculates displacement_counts and total_disp
        """
        park = Park([1] * self.m, self.n, circular = self.circular)

        for i in range(self.n**self.m):
            t = 0
            for j in range(self.m):
                self.disp_i[j][i] = park.displacement.count(j)
                self.wants_k[j][i] = park.pref(j+1)
                self.lel_i[j][i] = park.lel(j+1)
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
            self.passed_k[:,i] = park.passed
            park.next()

        self.total_disp = np.transpose(np.matrix.transpose(self.disp_i).dot(np.arange(self.m))) 

    #########################
    # Related stats to get  #
    #########################

    def get_graded(self, stat, grading = None):
        """
        TODO - test and integrate with plotting
        """
        if type(grading) == type(None):
            grading = self.defect
        
        categories = np.sort(np.unique(grading))
        labels = np.sort(np.unique(stat))

        cnts = np.zeros((len(categories), len(labels)))

        for i in range(len(stat)):
            cnts[np.searchsorted(categories, grading[i]),np.searchsorted(labels, stat[i])] += 1

        return categories, labels, cnts 
    
    
    @staticmethod
    def update_files():
        """runs the iterative function on all files"""
        for filename in os.listdir(IterateStats.save_dir):
            print(filename)
            IterateStats.update_file(filename)
    
    @staticmethod
    def update_file(filename):
        """
        updated specified file
        """
        f = os.path.join(IterateStats.save_dir, filename)
        if os.path.isfile(f):
            i = filename.index("n")
            j = filename.index("m")
            circ = True
            try:
                k = filename.index("c")
            except:
                circ = False
                k = filename.index(".")
    
            n = int(filename[i + 1:j])
            m = int(filename[j + 1:k])
            stats = IterateStats(n,m,circ)
            stats.iterate()
            stats.save()

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
    # METHODS FOR GTTING STATISTICS #
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

    def lel(self,i = 1):
        """
        returns number of cars whose prefered spot is the same as the ith car
        """
        return len(list(filter(lambda x: x.preference == self.cars[i-1].preference , self.cars)))

    def pref(self,i = 1):
        """
        returns number of cars whose prefered spot is i
        """
        return len(list(filter(lambda x: x.preference == i , self.cars)))

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

    @staticmethod
    def shuffle(p1, p2):
        """
        #TODO - useful for random_def
        Returns the shuffle of parking function 1 with parking function 2. 
        I believe shuffle was in both meyles,harris2023 and maybe another paper? 
        """

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

class IterDefect():
    def __init__(self, d, n, m = None):
        self.d = d
        self.n = n
        self.m = m or n
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.n <= 20:
            x = self.n
            self.n += 1
            return x
        else:
            raise StopIteration
