#!python
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False

import numpy as np
cimport numpy as np
import cython
cimport cython
import warnings
import heapq

# define consistent types to use everywhere
ctypedef np.int32_t int_t
ctypedef np.float32_t float_t
ctypedef np.npy_bool bool_t
ctypedef np.int64_t int64_t  # for searchsorted output

from libcpp.algorithm cimport sort
from libc.string cimport memcpy
from libc.stdint cimport int32_t

cdef extern from "stddef.h":
    ctypedef size_t size_t

from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector
from libcpp.utility cimport pair

# matching numpy types for python parts (np.int32, np.float32)
int_t_np = np.int32
float_t_np = np.float32

cdef extern from "<stdlib.h>":
    double drand48()

cdef float_t randnum() noexcept:
    return <float_t>drand48()

# LEF statuses
cdef int_t NUM_STATUSES = 5  # moving, paused, bound
cdef int_t STATUS_MOVING = 0  # LEF moved last time
cdef int_t STATUS_PAUSED = 1  # LEF failed to move the last step because it was paused
cdef int_t STATUS_STALLED = 2  # LEF failed to move the last step because it was stalled at another LEF or boundary
cdef int_t STATUS_CAPTURED = 3  # LEF is bound by CTCF and cannot move
cdef int_t STATUS_INACTIVE = 4  # A leg is inactive and cannot move

# occipied array statuses
cdef int_t OCCUPIED_FREE = -2
cdef int_t OCCUPIED_BOUNDARY = -1

# move policies
cdef int_t MOVE_POLICY_BOTH = 0  # both legs can move
cdef int_t MOVE_POLICY_ALTERNATE = 1  # only one leg can move at a time, alternating between legs with a given rate
cdef int_t MOVE_POLICY_ONELEG_RANDOM = 2  # only one leg can move at a time, and it's a random leg


cdef inline (int_t, int_t) find_adjacent_anchors(float_t x, int_t[::1] arr, int_t size) noexcept nogil:
    cdef int_t target = <int_t>x
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = size
    cdef Py_ssize_t mid

    while left < right:
        mid = (left + right) >> 1
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    cdef int_t insert = <int_t>left
    cdef int_t left_idx = insert - 1
    cdef int_t right_idx = insert
    if left_idx < 0:
        left_idx = -1
    if right_idx >= size:
        right_idx = -1

    return left_idx, right_idx


# simple binary search of existing value in an int_t array - assumes the value exists.
cdef inline int_t binary_search(int_t[::1] arr, int_t size, int_t target) noexcept nogil:
    cdef int_t left = 0
    cdef int_t right = size
    cdef int_t mid

    while left < right:
        mid = (left + right) >> 1
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


# a limitation of Cython - can't share constants between Cython and Python, so need to define them in Python as a dict
constants = {}
constants['NUM_STATUSES'] = NUM_STATUSES
constants['STATUS_MOVING'] = STATUS_MOVING
constants['STATUS_PAUSED'] = STATUS_PAUSED
constants['STATUS_STALLED'] = STATUS_STALLED
constants['STATUS_CAPTURED'] = STATUS_CAPTURED
constants['STATUS_INACTIVE'] = STATUS_INACTIVE
constants['OCCUPIED_FREE'] = OCCUPIED_FREE
constants['OCCUPIED_BOUNDARY'] = OCCUPIED_BOUNDARY
constants['MOVE_POLICY_BOTH'] = MOVE_POLICY_BOTH
constants['MOVE_POLICY_ALTERNATE'] = MOVE_POLICY_ALTERNATE
constants['MOVE_POLICY_ONELEG_RANDOM'] = MOVE_POLICY_ONELEG_RANDOM





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
        If True, the LEFs are not loaded at the start of the simulation. Default is False (load randomly).

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

    Public Methods
    --------------
    steps(step_start, step_end, watch=False)
        Perform a number of steps, optionally watching the positions of the LEFs
    set_watches(watch_array, max_events)
        Set the watches for the simulation
    get_events()
        Get the events that were triggered by watches
    get_occupied()
        Get the occupied positions
    get_LEFs()
        Get the positions of the LEFs
    get_statuses()
        Get the statuses of the LEFs
    force_load_LEFs(positions, statuses=None)
        Force the LEFs to specific positions with matching statuses (optional) - for debug/testing
    """
    cdef int_t N
    cdef int_t NLEFs
    # user defined arrays - unload (load is not needed because cumulatively loaded probabilities are used)
    cdef float_t [:, ::1] unload_prob
    # user defined arrays - CTCF interactions
    cdef float_t [:, ::1] capture_prob
    cdef float_t [::1] release_prob
    cdef float_t [::1] pause_prob
    # internal arrays
    cdef float_t [::1] load_prob_cumulative
    cdef int_t [:, ::1] LEFs
    cdef int_t [:, ::1] statuses
    cdef int_t [::1] occupied

    # load cache and related arrays
    cdef int_t load_cache_length
    cdef int_t load_cache_position
    cdef int_t [::1] load_pos_array

    # events and watches relsated stuff
    cdef int_t [:, ::1] events
    cdef int_t [::1] watch_mask
    cdef int_t event_number
    cdef int_t max_events

    # arrays needed for djikstra calculations
    cdef int_t [::1] lefs_pos_flat_sorted
    cdef int_t [:, ::1] lef_neigh_pos
    cdef float_t [:, ::1] lef_neigh_dist
    cdef float_t [::1] djikstra_dist

    # Policies and global attributes like probabilities
    cdef int_t move_policy
    cdef float_t alternate_prob

    def __init__(
        self,
        NLEFs,
        N,
        load_prob,
        unload_prob,
        capture_prob,
        release_prob,
        pause_prob,
        skip_load=False,
        move_policy=MOVE_POLICY_BOTH,
        alternate_prob=0.01,
    ):
        """
        Initialize the class with the probabilities of loading, unloading, capturing, releasing, and pausing
        """
        # safety checks so that we don't accidentally load/unload at the boundaries
        load_prob[0:2] = 0
        load_prob[len(load_prob) - 2 : len(load_prob)] = 0

        # cumulative load_prob arrays for cached load_prob function
        cum_load_prob = np.cumsum(load_prob)
        cum_load_prob = cum_load_prob / float(cum_load_prob[len(cum_load_prob) - 1])
        self.load_prob_cumulative = np.array(cum_load_prob, float_t_np, order="C")

        self.NLEFs = NLEFs
        self.N = N

        # check that all arrays are of the right size and shape
        if len(load_prob) != self.N:
            raise ValueError(f"Load probabilities must be of length {self.N}, not {len(load_prob)}")
        if len(load_prob.shape)!=1:
            raise ValueError(f"Load probabilities must be 1D, not {len(load_prob.shape)}D")
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

        # main arrays - ensure continuous
        self.capture_prob = np.array(capture_prob, order="C", dtype=float_t_np)
        self.release_prob = np.array(release_prob, order="C", dtype=float_t_np)
        self.unload_prob = np.array(unload_prob, order="C", dtype=float_t_np)
        self.pause_prob = np.array(pause_prob, order="C", dtype=float_t_np)

        self.LEFs = np.zeros((self.NLEFs, 2), dtype=int_t_np, order="C")
        self.statuses = np.full((self.NLEFs, 2), STATUS_MOVING, dtype=int_t_np, order="C")

        # some safety things for occupied array
        self.occupied = np.full(self.N, OCCUPIED_FREE, dtype=int_t_np, order="C")
        self.occupied[0] = OCCUPIED_BOUNDARY
        self.occupied[self.N - 1] = OCCUPIED_BOUNDARY

        # dijkstra related arrays - 2x NLEF long. Pre-allocating them now.
        self.lef_neigh_pos = np.zeros((self.NLEFs * 2, 3), dtype=int_t_np, order="C")
        self.lef_neigh_dist = np.zeros((self.NLEFs * 2, 3), dtype=float_t_np, order="C")
        self.lefs_pos_flat_sorted = np.zeros(self.NLEFs * 2, dtype=int_t_np, order="C")
        self.djikstra_dist = np.zeros(self.NLEFs * 2, dtype=float_t_np, order="C")

        # cache related things
        self.load_cache_length = 4096 * 4
        self.load_cache_position = 99999999

        if not skip_load:
            for ind in range(self.NLEFs):
                self.load_lef(ind)

        self.move_policy = move_policy
        self.alternate_prob = alternate_prob

    # Public functions first. They actually call private functions to do the work.
    def get_occupied(self):
        return np.array(self.occupied)

    def get_statuses(self):
        return np.array(self.statuses)

    def get_LEFs(self):
        return np.array(self.LEFs)

    def force_load_LEFs(self, positions, statuses=None):
        """
        A function used for testing: forces LEFs to specific positions with matching occupied array
        """
        cdef int_t lef, leg
        if positions.shape != (self.NLEFs, 2):
            raise ValueError("Positions must be of length NLEFs * 2")
        if statuses is None:
            statuses = np.full((self.NLEFs, 2), STATUS_MOVING, dtype=int_t_np, order="C")
        else:
            if statuses.shape != (self.NLEFs, 2):
                raise ValueError("Statuses must be of shape NLEFs x 2")
        for lef in range(self.NLEFs):
            for leg in range(2):
                self.LEFs[lef, leg] = positions[lef, leg]
                self.occupied[positions[lef, leg]] = lef + self.NLEFs * leg
                self.statuses[lef, leg] = statuses[lef, leg]

    def steps(self, step_start, step_end, watch=False):
        """
        Perform a number of steps, and watch the positions of the LEFs.
        This function optionally activates the watches.
        To use watches, first call set_watches(watch_array, max_events) to set the watches.
        Then simulate a block of steps and call get_events() to get the events.
        """
        cdef int_t i
        cdef bool_t watch_flag = watch
        for i in range(step_start, step_end):
            self.unload()
            if self.move_policy == MOVE_POLICY_ALTERNATE:
                self.alternate_legs()
            self.step()
            if watch_flag:
                self.watch(i)

    def alternate_legs(self):
        """
        A function to alternate the legs that can move.

        * If one of the legs is CTCF bound, then the other will be assigned to move
        * If one of the legs is inactive, then with a given probability it activates and the other becomes inactive

        A choice of one leg is moving when the other is on a CTCF makes sense.
        It allows for faster exploration and we talked about it.
        However, the question was what to do with stalled legs.
        But here we enter an unpleasant problem of how do legs resolve collisions.

        There is a world in which a LEF is stalled at another LEF, and keeps trying to pass through it
        while also being more vulnerable to unloading.
        Imagine that we assigned the non-stalled leg to be active, and the stalled leg is inactive.
        Is a LEF still vulnerable to unloading? Is it still stalled?
        If the stalled leg is inactive, then a LEF (in an actual physical world, a molecule of cohesin)
        won't be trying to move in the direction of the cohesin it is stalled against.
        Would it be more likely to be unloaded then?
        If we want increased unloading, we have to assume it "keeps trying", and is more vulnerable to unloading.
        """
        cdef int_t lef
        cdef bool_t leg0_bound, leg1_bound, leg0_inactive, leg1_inactive
        for lef in range(self.NLEFs):
            leg0_bound = (self.statuses[lef, 0] == STATUS_CAPTURED)  # not actually touching stalled
            leg1_bound = (self.statuses[lef, 1] == STATUS_CAPTURED)
            leg0_inactive = (self.statuses[lef, 0] == STATUS_INACTIVE)
            leg1_inactive = (self.statuses[lef, 1] == STATUS_INACTIVE)
            if leg0_bound & leg1_bound:
                continue  # both legs bound, nothing to do
            elif leg0_bound | leg1_bound:  # one leg can't move - another can't be inactive
                if leg0_inactive:
                    self.statuses[lef, 0] = STATUS_MOVING
                if leg1_inactive:
                    self.statuses[lef, 1] = STATUS_MOVING
                continue  # one leg bound - no swapping or reassignment, continuing
            else:  # none of the legs are bound (captured that is)
                # This will happen e.g. if one leg became released, or at the start
                if not (leg1_inactive | leg0_inactive):  # one leg has to be made inactive
                    self.statuses[lef, 0 if randnum() > 0.5 else 1] = STATUS_INACTIVE
            if randnum() < self.alternate_prob:
                # swapping of active and inactive legs with some probability
                if leg0_inactive:
                    self.statuses[lef, 0] = STATUS_MOVING
                    self.statuses[lef, 1] = STATUS_INACTIVE
                if leg1_inactive:
                    self.statuses[lef, 1] = STATUS_MOVING
                    self.statuses[lef, 0] = STATUS_INACTIVE

    def set_watches(self, watch_array, max_events=100000):
        """
        Set the watches for the simulation.
        The watches are positions in the array that trigger an event when both legs of a LEF are at watched positions.
        The events are stored in the events array, which can be accessed with get_events().

        Parameters:
        watch_array : list-like or array-like
            An array or list containing positions to watch.
        max_events : int
            The maximum number of events to store.
        """
        # Initialize watches to a zeroed array of size N, where each index represents a position in the simulation.
        self.watch_mask = np.zeros(self.N, dtype=int_t_np, order="C")

        # Set the watches at specified positions.
        for position in watch_array:
            if position < self.N and position >= 0:  # Ensure the position is within bounds.
                self.watch_mask[position] = 1
            else:
                raise ValueError("Watch position is out of bounds.")

        # Initialize the events array to store events. Each event records the position of both legs and the time.
        self.events = np.zeros((max_events, 3), dtype=int_t_np, order="C")  # Each event stored as [pos1, pos2, time].
        self.event_number = 0  # Reset the event number counter.
        self.max_events = max_events  # Store the maximum events allowed

    def get_events(self, reset=False):
        ar = np.array(self.events)
        event_num = self.event_number  # cache event number
        if reset:
            self.event_number = 0
        return ar[:event_num]

    # Internal functions next. They are called by the public functions to do the actual logic of the simulation.
    cdef watch(self, int_t time):
        """
        An internal method to trigger events when both legs are at a watched position.
        """
        cdef int_t lef
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
        cdef int_t pos, leflen, leg

        while True:
            pos = self.get_cached_load_position()
            if pos >= self.N - 2 or pos <= 1:  # N-1 is a boundary, we need to be N-4 to fit a 2-wide LEF
                warnings.warn(f"Ignoring load_prob at 0 or end. load_prob at: {pos}")
                continue

            # checking all 3 positions for consistency and to avoid a LEF being born around another LEF's leg
            if (
                self.occupied[pos - 1] != OCCUPIED_FREE
                or self.occupied[pos] != OCCUPIED_FREE
                or self.occupied[pos + 1] != OCCUPIED_FREE
            ):
                continue

            # Need to make LEFs of different sizes - 1 or 2 wide, to avoid checkering in the contact map
            leflen = 2 if randnum() > 0.5 else 1  # 1 or 2 wide LEF at loading
            for leg in range(2):
                self.LEFs[lef, leg] = pos - 1 + leg * leflen  # [pos-1, pos] or [pos-1, pos+1]
                self.statuses[lef, leg] = STATUS_MOVING
                self.occupied[pos - 1 + leg * leflen] = lef + self.NLEFs * leg  # record which LEF/leg is there
            break

    cdef unload(self):
        """ An internal method to try to unload all the LEFs - is called by "step" method"""
        cdef int_t lef, leg, s1, s2
        cdef float_t unload, unload1, unload2

        for lef in range(self.NLEFs):  # check all LEFs
            s1 = self.statuses[lef, 0]
            s2 = self.statuses[lef, 1]
            unload1 = self.unload_prob[self.LEFs[lef, 0], self.statuses[lef, 0]]
            unload2 = self.unload_prob[self.LEFs[lef, 1], self.statuses[lef, 1]]

            # logic for releaseing - subject to change
            if s1 == s2:  # same statuses for both legs
                unload = (unload1 + unload2) / 2  # take the mean of probabilities - each leg unloads "independently"

            # This is the only exception, and that is because CTCF protects "the whole thing", not just one leg
            elif s1 == STATUS_CAPTURED or s2 == STATUS_CAPTURED:  # one leg is at CTCF, another is not
                unload = min(unload1, unload2)  # take the most protective probability - smallest unload

            # Treating paused and stalled the same way - each leg is independent
            # one leg stalled another moving or paused or inactive
            # It may be possible to treat inactive legs differently - not implemented. 
            # Should be considered with the first realistic use case 
            elif s1 == STATUS_STALLED or s2 == STATUS_STALLED or s1 == STATUS_INACTIVE or s2 == STATUS_INACTIVE:
                unload = (unload1 + unload2) / 2  # take the mean, higher prob if stalled leg is unloaded faster
            elif s1 == STATUS_PAUSED or s2 == STATUS_PAUSED:  # one leg paused another moving
                unload = (unload1 + unload2) / 2  # take the mean, higher prob if paused leg is unloaded faster
            else:
                raise ValueError(f"statuses are not consistent: {s1} and {s2}")

            if randnum() < unload:
                for leg in range(2):  # statuses are re-initialized in load, occupied here
                    self.occupied[self.LEFs[lef, leg]] = OCCUPIED_FREE
                self.load_lef(lef)

    cdef int_t get_cached_load_position(self):
        """
        An internal method to get a cached load position.
        This is necessary because the load position is obtained by binary search,
        and we don't want to call np.searchsorted() every time.
        """

        if self.load_cache_position >= self.load_cache_length - 1:
            foundArray = np.array(
                np.searchsorted(
                    self.load_prob_cumulative, np.random.random(self.load_cache_length)
                ),
                dtype=int_t_np,
                order="C",
            )
            self.load_pos_array = foundArray
            self.load_cache_position = -1
        self.load_cache_position += 1
        return self.load_pos_array[self.load_cache_position]

    cdef step(self):
        """An internal (= C++, not Python) method to perform a step
        It is called by steps() and steps_watch() methods.
        It does the following logic:
        1. Check if the LEF can be captured or released by a CTCF
        2. Check if the LEF can move
        3. Move the LEF if it can
        """
        cdef int_t lef, leg, pos, newpos, leg_ind
        cdef bool_t swap_order
        for lef in range(self.NLEFs):
            swap_order = randnum() < 0.5  # pick a random starting leg
            for leg_ind in range(2):
                # go through two legs in random order
                leg = leg_ind if swap_order else 1 - leg_ind

                # capture and release logic goes here - it s simple
                if randnum() < self.capture_prob[self.LEFs[lef, leg], leg]:  # try to capture the leg
                    # if it's not inactive we can capture it
                    if self.statuses[lef, leg] != STATUS_INACTIVE:
                        self.statuses[lef, leg] = STATUS_CAPTURED
                if randnum() < self.release_prob[self.LEFs[lef, leg]]:  # try to release the leg
                    if self.statuses[lef, leg] == STATUS_CAPTURED:  # if it's captured we can release it
                        self.statuses[lef, leg] = STATUS_MOVING

                # moving logic is somewhat more complicated
                pos = self.LEFs[lef, leg]
                # check if "this" leg is captured or inactive - attempt a move i fit's not
                if  self.statuses[lef, leg] != STATUS_CAPTURED and self.statuses[lef, leg] != STATUS_INACTIVE:
                    newpos = pos + (2 * leg - 1)  # leg 1 moves "right" - increasing numbers
                    if self.occupied[newpos] == OCCUPIED_FREE:  # Can we go there?
                        if randnum() > self.pause_prob[pos]:  # check if we are paused
                            # The leg can move, so we need to update the arrays
                            self.occupied[newpos] = lef + self.NLEFs * leg  # update occupied array
                            self.occupied[pos] = OCCUPIED_FREE  # free the old position
                            self.LEFs[lef, leg] = newpos  # update position of leg
                            self.statuses[lef, leg] = STATUS_MOVING  # we are moving now!
                            # for policy ALTERNATE we don't need this, because one of the two legs is inactive or bound.
                            if self.move_policy == MOVE_POLICY_ONELEG_RANDOM:
                                break  # this leg moved, we don't attempt the other leg
                        else:  # we are paused - can't move
                            self.statuses[lef, leg] = STATUS_PAUSED  # we are paused because of the pause probability
                    else:  # we are stalled - can't move
                        self.statuses[lef, leg] = STATUS_STALLED  # we are stalled because other position is occupied

    def compute_pair_distances(self,
                            float_t start,
                            float_t end
                            ):
        """
        Compute the distances between two  positions using Dijkstra's algorithm.

        IMPORTANT: need to call populate_dijkstra_arrays before calling this function

        Parameters
        ----------
        start : float
            The start position
        end : float
            The end position

        Returns
        -------
        float
            The distance between the two positions through the LEF graph
        """

        
        cdef int_t size = self.lefs_pos_flat_sorted.shape[0]
        cdef float_t INF = 1e10
        cdef int_t i, i_start_left, i_start_right, i_end_left, i_end_right
        cdef float_t direct_dist, best_anchor_bridge
        cdef float_t dist_start_left, dist_start_right, dist_end_left, dist_end_right
        cdef float_t  anchor_end_dist, new_dist, cur_dist, best_dist
        cdef int_t u, v, k
        cdef int_t[::1] pos_view = self.lefs_pos_flat_sorted  # shortcut for simplicity of code
        cdef priority_queue[pair[float_t, int_t]] pq
        
        direct_dist = abs(end - start)
        # find the closest anchors to the start and end
        i_start_left, i_start_right = find_adjacent_anchors(start, pos_view, size)
        i_end_left, i_end_right = find_adjacent_anchors(end, pos_view, size)

        # calculate the distances to the start and end anchors
        dist_start_left = (start - pos_view[i_start_left]) if i_start_left != -1 else INF
        dist_start_right = (pos_view[i_start_right] - start) if i_start_right != -1 else INF
        dist_end_left = (end - pos_view[i_end_left]) if i_end_left != -1 else INF
        dist_end_right = (pos_view[i_end_right] - end) if i_end_right != -1 else INF

        # check if we can use direct distance and avoid djikstra
        best_anchor_bridge = min(dist_start_left + dist_end_left,
                                dist_start_left + dist_end_right,
                                dist_start_right + dist_end_left,
                                dist_start_right + dist_end_right)        
        if direct_dist <= best_anchor_bridge:
            return direct_dist            

        # initialize the distances from each anchor to INF
        for k in range(size):
            self.djikstra_dist[k] = INF            

        # initialize the distances from the start to the start anchors
        # This is basically "two searches in one" - from the anchors on the two sides of the start
        if i_start_left != -1 and dist_start_left < self.djikstra_dist[i_start_left]:
            self.djikstra_dist[i_start_left] = dist_start_left
            pq.push(pair[float_t,int_t](-dist_start_left, i_start_left))

        if i_start_right != -1 and dist_start_right < self.djikstra_dist[i_start_right]:
            self.djikstra_dist[i_start_right] = dist_start_right
            pq.push(pair[float_t,int_t](-dist_start_right, i_start_right))

        # initialize the best distance to the direct distance
        best_dist = direct_dist

        while not pq.empty():  # standard djiikstra - get the closest node
            top = pq.top()
            cur_dist = -top.first  # minus because we use a max heap
            u = top.second
            pq.pop()

            if cur_dist > self.djikstra_dist[u]:
                continue  # we already have a better distance
            if cur_dist > best_dist:
                break  # we are further than direct distance - exiting

            # The array djikstra_dist keeps distances from the start to the current node
            # But best_dist is the best distance from the start to the end
            # we compare cur_dist + ancor_end_dist to best_dist to see if we can exit early
            # But when we update best_dist, we track distances from start to the current node, not the end
            anchor_end_dist = abs(pos_view[u] - end)  # distance from current view to the end
            if cur_dist + anchor_end_dist < best_dist:  # we reach the end faster than current best distance
                best_dist = cur_dist + anchor_end_dist  # update the best distance

            for k in range(3):  # iterating over 3 neighbors - left, right, shortcut
                v = self.lef_neigh_pos[u, k]
                new_dist = cur_dist + self.lef_neigh_dist[u, k]
                if new_dist < self.djikstra_dist[v] and new_dist < best_dist:  # we have a better distance
                    self.djikstra_dist[v] = new_dist
                    pq.push(pair[float_t,int_t](-new_dist, v))  # push the new distance to the queue

        return min(direct_dist, best_dist) # should always be the best_dist, but just in case
        

    def populate_dijkstra_arrays(self, float_t lef_length):
        """
        Populate arrays used by Dijkstra. Called once before computing distances.

        This function populates the following arrays: lefs_pos_flat_sorted, lef_neigh_pos, lef_neigh_dist.
        Each LEF has 3 neighbors: the previous LEF, the next LEF, and a shortcut to the other leg of the same LEF.

        Parameters
        ----------
        lef_length : float
            The length of a LEF

        Arrays
        ------
        lefs_pos_flat_sorted : shape (NLEFs * 2), int
            The positions of the LEFs sorted in ascending order
        lef_neigh_pos : shape (NLEFs * 2, 3), int
            The positions of the neighbors of each LEF
        lef_neigh_dist : shape (NLEFs * 2, 3), float
            The distances to the neighbors of each LEF
        """
        cdef int_t i, lef_ind, lef, leg, occval, pos, pos2, lef_ind2
        cdef float_t dist
        cdef int_t size = self.NLEFs * 2        

        # flatten LEF positions with memcpy, sort with std sort
        memcpy(&self.lefs_pos_flat_sorted[0], &self.LEFs[0,0], size * sizeof(int_t))
        sort(&self.lefs_pos_flat_sorted[0], &self.lefs_pos_flat_sorted[0] + size)

        # set direct neighbors
        for i in range(size):            
            # prev neighbor
            if i > 0:
                self.lef_neigh_pos[i, 0] = i - 1
                self.lef_neigh_dist[i, 0] = self.lefs_pos_flat_sorted[i] - self.lefs_pos_flat_sorted[i - 1]
            else:
                self.lef_neigh_pos[i, 0] = i
                self.lef_neigh_dist[i, 0] = 1e6  # large distance to avoid going back to yourself

            # next neighbor
            if i < size - 1:
                self.lef_neigh_pos[i, 1] = i + 1
                self.lef_neigh_dist[i, 1] = self.lefs_pos_flat_sorted[i + 1] - self.lefs_pos_flat_sorted[i]
            else:
                self.lef_neigh_pos[i, 1] = i
                self.lef_neigh_dist[i, 1] = 1e6

        # shortcuts: each LEF creates a bidirectional shortcut between its two legs
        for lef_ind in range(size):
            # going "direct" through occupied array            
            pos = self.lefs_pos_flat_sorted[lef_ind]
            occval = self.occupied[pos]
            lef = occval % self.NLEFs  # LEF index
            leg = occval // self.NLEFs  # leg index
            # find the position of the other leg
            pos2 = self.LEFs[lef, 1 - leg]
            # find the index of the other leg in the sorted array
            lef_ind2 = binary_search(self.lefs_pos_flat_sorted, size, pos2)            
            # set the shortcut
            self.lef_neigh_pos[lef_ind, 2] = lef_ind2
            self.lef_neigh_dist[lef_ind, 2] = lef_length
        

    def debug_get_djiikstra_arrays(self):
        """
        A debug function to get the Dijkstra arrays for visualization
        """
        return np.array(self.lefs_pos_flat_sorted), np.array(self.lef_neigh_pos), np.array(self.lef_neigh_dist), np.array(self.djikstra_dist)
        