#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True


import numpy as np
cimport numpy as np

import cython
cimport cython

cdef extern from "<stdlib.h>":
    double drand48()   

cdef cython.double randnum():
    return drand48()

# LEF statuses 
cdef int NUM_STATUSES = 3 # moving, paused, bound
cdef int STATUS_MOVING = 0  # LEF moved last time
cdef int STATUS_PAUSED = 1  # LEF failed to move the last step 
cdef int STATUS_CAPTURED = 2   # LEF is bound by CTCF and cannot move 
# Could add more statuses e.g. the leg is "resting" because only one leg can move at a time 
# but this is the case for a more complicated thing

# occipied array statuses 
cdef int OCCUPIED_FREE = -2 
cdef int OCCUPIED_BOUNDARY = -1 

# a limitation of Cython - we cannot share constants between Cython and Python, so we need to define them in Python as a dict
constants = {}
constants['NUM_STATUSES'] = NUM_STATUSES
constants['STATUS_MOVING'] = STATUS_MOVING
constants['STATUS_PAUSED'] = STATUS_PAUSED
constants['STATUS_CAPTURED'] = STATUS_CAPTURED
constants['OCCUPIED_FREE'] = OCCUPIED_FREE
constants['OCCUPIED_BOUNDARY'] = OCCUPIED_BOUNDARY



cdef class LEFSimulator(object):
    """
    A class to simulate a translocator with two legs that can capture to CTCF sites

    Parameters
    ----------
    NLEFs : int
        The number of LEFs in the system
    N : int
        The number of positions in the system
    load_prob : array-like (N)
        An array of probabilities of loading a LEF at each position
    unload_prob : array-like (N, num_statuses)
        An array of probabilities of unloading a LEF at each position and status
    capture_prob : array-like (N, 2)
        An array of probabilities of captureing a LEF to a CTCF site at each position
    release_prob : array-like (N)
        An array of probabilities of releaseing a LEF from a CTCF site at each position
    pause_prob : array-like (N)
        An array of probabilities of pausing a LEF at each position
    skip_load : bool
        If True, the LEFs are not loaded at the start of the simulation. Default is False (load LEFs randomly at the start of the simulation)

    Attributes
    ----------
    # excluding user-defined arrays 
    N : int
        The number of positions in the array
    NLEFs : int
        The number of LEFs in the system
    LEFs : array-like
        An array of the positions of the LEFs  (NLEFs x 2)
    statuses : array-like
        An array of the statuses of the LEFs (NLEFs x 2)
    occupied : array-like
        An array of the occupied positions (N)
    events : array-like
        An array of the events that were triggered by watches
    watches : array-like
        An array of the positions that are watched

    Methods
    -------
    steps(step_start, step_end)
        Perform a number of steps
    steps_watch(step_start, step_end)
        Perform a number of steps and watch the positions of the LEFs
    set_watches(watch_array, max_events)
        Set the watches for the simulation
    get_events()
        Get the events that were triggered by watches
    get_occupied()
        Get the occupied positions
    get_LEFs()
        Get the positions of the LEFs
    

    """
    cdef int N
    cdef int NLEFs
    # user defined arrays - unload (load is not needed because cumulatively loaded probabilities are used)
    cdef cython.double [:,:] unload_prob 
    # user defined arrays - CTCF interactions
    cdef cython.double [:, :] capture_prob
    cdef cython.double [:] release_prob    
    cdef cython.double [:] pause_prob
    # internal arrays
    cdef cython.double [:] load_prob_cumulative
    cdef np.int64_t [:, :] LEFs
    cdef np.int64_t [:, :] statuses 
    cdef np.int64_t [:] occupied 
    
    cdef int load_cache_length
    cdef int load_cache_position
    cdef np.int64_t [:] load_pos_array
    
    cdef np.int64_t [:, :] events
    cdef np.int64_t [:] watch_mask
    cdef np.int64_t event_number
    cdef np.int64_t max_events
    
    def __init__(self, NLEFs, N,  load_prob, unload_prob, capture_prob, release_prob, pause_prob, skip_load = False):
        """
        Initialize the class with the probabilities of loading, unloading, captureing, releaseing, and pausing
        """
        # safety checks so that we don't accidentally load/unload at the boundaries
        load_prob[0:2] = 0
        load_prob[len(load_prob)-2:len(load_prob)] = 0
         
        # cumulative load_prob arrays for cached load_prob function
        cumem = np.cumsum(load_prob)
        cumem = cumem / float(cumem[len(cumem)-1])
        self.load_prob_cumulative = np.array(cumem, np.double)
        
        self.NLEFs = NLEFs
        self.N = N 

        # check that all arrays are of the right size and shape        
        if len(load_prob) != self.N:
            raise ValueError(f"Load probabilities must be of length {self.N}, not {len(load_prob)}")
        if len(unload_prob) != self.N:
            raise ValueError(f"Unload probabilities must be of length {self.N}, not {len(unload_prob)}")
        if len(capture_prob) != self.N:
            raise ValueError(f"capture probabilities must be of length {self.N}, not {len(capture_prob)}")
        if len(release_prob) != self.N:
            raise ValueError(f"release probabilities must be of length {self.N}, not {len(release_prob)}")
        if len(pause_prob) != self.N:
            raise ValueError(f"Pause probabilities must be of length {self.N}, not {len(pause_prob)}")
        if len(unload_prob[0]) != NUM_STATUSES:
            raise ValueError(f"Unload probabilities must have {NUM_STATUSES} statuses, not {len(unload_prob[0])}")
        if len(capture_prob[0]) != 2:
            raise ValueError("capture probabilities must have 2 legs")
        
        # main arrays
        self.capture_prob = capture_prob
        self.release_prob = release_prob
        self.unload_prob = unload_prob
        self.pause_prob = pause_prob

        self.LEFs = np.zeros((self.NLEFs, 2), int)
        self.statuses = np.full((self.NLEFs, 2), STATUS_MOVING, int)

        # some safety things for occupied array
        self.occupied = np.full(self.N, OCCUPIED_FREE, dtype=int)        
        self.occupied[0] = OCCUPIED_BOUNDARY
        self.occupied[self.N - 1] = OCCUPIED_BOUNDARY
        
        self.load_cache_length = 4096 * 4
        self.load_cache_position = 99999999
        if not skip_load:
            for ind in range(self.NLEFs):
                self.load_lef(ind)


    def steps(self,step_start, step_end):
        """
        Perform a number of steps. This is a Python function to be called, rather than step() which is internal.         
        """
        cdef int i 
        for i in range(step_start, step_end):
            self.unload()
            self.step()
            
    def get_occupied(self):
        return np.array(self.occupied)
    
    def get_statuses(self):
        return np.array(self.statuses)
    
    def get_LEFs(self):
        return np.array(self.LEFs)

    def force_load_LEFs(self, positions):
        """
        A function used for testing: forces LEFs to specific positions with matching occupied array
        """
        cdef int lef, leg
        for lef in range(self.NLEFs):
            for leg in range(2):
                self.LEFs[lef, leg] = positions[lef, leg]
                self.occupied[positions[lef, leg]] = lef + self.NLEFs * leg
                self.statuses[lef, leg] = STATUS_MOVING

                    

    def steps_watch(self,step_start, step_end):
        """
        Perform a number of steps, and watch the positions of the LEFs.
        This function also activates the watches. 
        To use watches, first call set_watches(watch_array, max_events) to set the watches. 
        Then simulate a block of steps and call get_events() to get the events.
        """
        cdef int timestep
        for timestep in range(step_start, step_end):
            self.unload()
            self.step()
            self.watch(timestep)

    def set_watches(self, watch_array, max_events):
        """
        Set the watches for the simulation.
        The watches are positions in the array that trigger an event when both legs of a LEF are at the watched position.
        The events are stored in the events array, which can be accessed with get_events().
        
        Parameters:
        watch_array : list-like or array-like
            An array or list containing positions to watch.
        max_events : int
            The maximum number of events to store.
        """
        # Initialize watches to a zeroed array of size N, where each index represents a position in the simulation.
        self.watch_mask = np.zeros(self.N, dtype=np.int64) 

        # Set the watches at specified positions.
        for position in watch_array:
            if position < self.N and position >= 0:  # Ensure the position is within bounds.
                self.watch_mask[position] = 1
            else: 
                raise ValueError("Watch position is out of bounds.")

        # Initialize the events array to store events. Each event records the position of both legs and the time.
        self.events = np.zeros((max_events, 3), dtype=np.int64)  # Each event is stored as [pos1, pos2, time].
        self.event_number = 0  # Reset the event number counter.
        self.max_events = max_events  # Store the maximum events allowed


        def get_events(self): 
            ar = np.array(self.events)
            return ar[:self.max_events]  

        
    cdef watch(self, time):
        """
        An internal method to watch the positions of the LEFs and trigger events when both legs are at a watched position.
        """
        cdef int lef
        for lef in range(self.NLEFs):            
            if self.watch_mask[self.LEFs[lef, 0]] == 1 and self.watch_mask[self.LEFs[lef, 1]] == 1:
                self.events[self.event_number, 0] = self.LEFs[lef, 0]
                self.events[self.event_number, 1] = self.LEFs[lef, 1]
                self.events[self.event_number, 2] = time 
                self.event_number += 1 
            if self.event_number == self.max_events:
                raise ValueError("Events are full - increase max_events")            


    cdef load_lef(self, cython.int lef):
        """
        An internal method to load a given LEF - is called by "unload" method on its own
        """
        cdef int pos, leflen, leg
  
        while True:
            pos = self.get_cached_load_position()
            if pos >= self.N - 2 or pos <= 1:  # N-1 is a boundary, we need to be N-4 to fit a 2-wide LEF
                print("Ignoring load_prob at 0 or end. load_prob at:", pos)
                continue 
            
            if (self.occupied[pos-1] != OCCUPIED_FREE) | (self.occupied[pos] != OCCUPIED_FREE) | (self.occupied[pos+1] != OCCUPIED_FREE):
                continue  # checking all 3 positions for consistency and to avoid a LEF being born around another LEF's leg

            # Need to make LEFs of different sizes - 1 or 2 wide, to avoid checkering in the contact map
            leflen = 2 if randnum() > 0.5 else 1  # 1 or 2 wide LEF at loading
            for leg in range(2):
                self.LEFs[lef, leg] = pos - 1 + leg * leflen  #[pos-1, pos] or [pos-1, pos+1] 
                self.statuses[lef, leg] = STATUS_MOVING
                self.occupied[pos - 1 + leg * leflen] = lef + self.NLEFs *leg  # record which LEF is there and which leg            
            return

    cdef unload(self):
        cdef int lef, leg, s1, s2
        cdef double unload, unload1, unload2
         
        for lef in range(self.NLEFs):     
            s1 = self.statuses[lef, 0]
            s2 = self.statuses[lef, 1]            
            unload1 = self.unload_prob[self.LEFs[lef, 0], self.statuses[lef,0]]
            unload2 = self.unload_prob[self.LEFs[lef, 1], self.statuses[lef,1]]

            # logic for releaseing - subject to change 
            if s1 == s2:  # same statuses for both legs 
                unload = (unload1 + unload2) / 2   # just take the mean of probabilities - it's fair
            elif s1 == STATUS_CAPTURED or s2 == STATUS_CAPTURED:  # one leg is at CTCF, another is not
                # This is the only exception, and that is because CTCF protects "the whole thing", not just one leg
                unload = min(unload1, unload2)  # take the most protective probability - smallest unload
            elif s1 == STATUS_PAUSED or s2 == STATUS_PAUSED:  # one leg paused another moving 
                unload = (unload1+unload2) / 2  # take the mean, which means higher prob if stalled leg is 
            else:
                raise ValueError("Today 2+2 = -5e452, the number of atoms in the universe is negative, and the bugs are all out.")
            
            if randnum() < unload:
                for leg in range(2): # statuses are re-initialized in load, occupied here
                    self.occupied[self.LEFs[lef, leg]] = OCCUPIED_FREE
                self.load_lef(lef)
    
    cdef int get_cached_load_position(self):
        """
        An internal method to get a cached load position. 
        This is necessary because the load position is drawn from a distribution, and we don't want to call np.random.random() every time.
        """
    
        if self.load_cache_position >= self.load_cache_length - 1:
            foundArray = np.array(np.searchsorted(self.load_prob_cumulative, np.random.random(self.load_cache_length)), dtype = np.int64)
            self.load_pos_array = foundArray
            self.load_cache_position = -1        
        self.load_cache_position += 1         
        return self.load_pos_array[self.load_cache_position]    

    cdef step(self):
        """An internal (= C++, not Python) method to perform a step
        It is called by steps() and steps_watch() methods.
        It does the following logic: 
        1. Check if the LEF can capture or release to a CTCF site
        2. Check if the LEF can move
        3. Move the LEF if it can
        """
        cdef int lef, leg        
        for lef in range(self.NLEFs):
            for leg in range(2):                
                if randnum() < self.capture_prob[self.LEFs[lef, leg], leg]: # try to capture the leg 
                    self.statuses[lef, leg] = STATUS_CAPTURED
                if randnum() < self.release_prob[self.LEFs[lef, leg]]: # try to release the leg
                    self.statuses[lef, leg] = STATUS_MOVING  # We are now moving

                pos = self.LEFs[lef, leg]
                if self.statuses[lef, leg] != STATUS_CAPTURED: # not bound = can move
                    newpos = pos + (2 * leg - 1)  # leg 1 moves "right" - increasing numbers
                    if (self.occupied[newpos] == OCCUPIED_FREE) and (randnum() > self.pause_prob[pos]) : #can move and are not paused
                        self.occupied[newpos] = lef + self.NLEFs * leg  # update occupied array 
                        self.occupied[pos] = OCCUPIED_FREE  # free the old position
                        self.LEFs[lef, leg] = newpos   # update position of leg 
                        self.statuses[lef,leg] = STATUS_MOVING  # we are moving now!    
                    else:
                        self.statuses[lef,leg] = STATUS_PAUSED  # we are paused now!             