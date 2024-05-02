
import pytest
import numpy as np 

from lefs_cython.simple import LEFSimulator, constants
def test_load_and_translocation():
    """
    Test if
    - LEF is loaded at the correct position
    - LEF moves to the correct position (given stochasticity of loading)
    """
    N_LEFS = 1 
    N = 100 

    load_array = np.zeros(N)
    load_array[40] = 1 

    unload_array = np.zeros((N, 3))  # no unloading

    capture_array = np.zeros((N, 2))  # no CTCF
    release_array = np.zeros(N) 

    pause_array = np.zeros(N)  # no pausing
    right_positions = []
    for attempt in range(20):
        LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array)

        LEF.steps(0, 5)
        pos = LEF.get_LEFs()

        # loading was done at (39, 40) or at (39, 41) 
        # LEF should have moved 5 steps (left leg left, right leg right) 
        assert pos[0,0] == 34 
        assert pos[0,1] == 45 or pos[0,1] == 46
        right_positions.append(pos[0,1])
    assert len(set(right_positions)) == 2  # LEF was loaded at (39, 40) or at (39, 41)




def test_translocate_single():
    """
    Test if LEF translocates deterministically
    """
    
    N_LEFS = 1 
    N = 100 

    load_array = np.zeros(N)
    load_array[40] = 1 

    unload_array = np.zeros((N, 3))  # no unloading

    capture_array = np.zeros((N, 2))  # no CTCF
    release_array = np.zeros(N) 

    pause_array = np.zeros(N)  # no pausing

    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True) 
    LEF.force_load_LEFs(np.array([[40, 41]]))
    LEF.steps(0, 5)
    pos = LEF.get_LEFs()
    assert pos[0,0] == 35
    assert pos[0,1] == 46


def test_collide_two_lefs(): 
    """
    Test if two LEFs collide. 
    * Force-load two LEFs at given positions 
    * Wait for enough steps for them to collide
    * Check if they collided
    """
    N_LEFS = 2 
    N = 100 

    load_array = np.zeros(N)
    load_array[40] = 1 
    load_array[41] = 1 

    unload_array = np.zeros((N, 3))  # no unloading

    capture_array = np.zeros((N, 2))  # no CTCF
    release_array = np.zeros(N) 

    pause_array = np.zeros(N)  # no pausing

    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True) 
    LEF.force_load_LEFs(np.array([[40, 41], [50, 51]]))
    LEF.steps(0, 10)
    pos = LEF.get_LEFs()
    assert pos[0,0] == 30
    assert pos[0,1] == 45
    assert pos[1,0] == 46
    assert pos[1,1] == 61


def test_occupied_array_matches(): 
    """
    Test if the occupied array matches the LEF positions in a LEFs with and without collision 
    """
    N_LEFS = 3
    N = 100 

    load_array = np.zeros(N)
    load_array[40] = 1 
    load_array[41] = 1 

    unload_array = np.zeros((N, 3))  # no unloading

    capture_array = np.zeros((N, 2))  # no CTCF
    release_array = np.zeros(N) 

    pause_array = np.zeros(N)  # no pausing

    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True) 
    LEF.force_load_LEFs(np.array([[40, 41], [50, 51], [75, 76]]))
    LEF.steps(0, 10)
    pos = LEF.get_LEFs()
    occ = LEF.get_occupied()
    
    # iterate over positions and check that occupied array is what it is supposed to be
    for lef in range(N_LEFS):
        for leg in range(2):
            assert occ[pos[lef, leg]] == lef  + N_LEFS * leg
    # check it manually too (mostly for the person reading this to understand the logic)
    assert occ[30] == 0  # zeroth LEf zeroth leg 
    assert occ[45] == 3  # zeroth LEF first leg  (NLEFs * leg + LEF index)
    assert occ[46] == 1  # first LEF first leg
    assert occ[61] == 4  # first LEF second leg
    assert occ[65] == 2  # third LEF first leg
    assert occ[86] == 5  # third LEF second leg

    # check boundaries
    assert occ[0] == constants["OCCUPIED_BOUNDARY"]
    assert occ[-1] == constants["OCCUPIED_BOUNDARY"]

    # all other positions should be OCCIPUED_FREE 
    for i in range(1, N-1):
        if i not in [30, 45, 46, 61, 65, 86]:
            assert occ[i] == constants["OCCUPIED_FREE"]

def test_status_paused_moving(): 
    """
    Load two LEFs,
    let them make a few steps and verify that they are moving 
    let them collide, verify that collided legs are STATUS_PAUSED 
    Let them continue until two other legs hit wall and verify that now all legs are STATUS_PAUSED
    """

    N_LEFS = 2
    N = 100 

    load_array = np.zeros(N)
    load_array[40] = 1 
    load_array[41] = 1 

    unload_array = np.zeros((N, 3))  # no unloading

    capture_array = np.zeros((N, 2))  # no CTCF
    release_array = np.zeros(N) 

    pause_array = np.zeros(N)  # no pausing

    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True) 
    LEF.force_load_LEFs(np.array([[40, 41], [50, 51]]))
    LEF.steps(0, 2) 
    status = LEF.get_statuses()
    assert status[0, 0] == constants["STATUS_MOVING"]
    assert status[0, 1] == constants["STATUS_MOVING"]
    assert status[1, 0] == constants["STATUS_MOVING"]
    assert status[1, 1] == constants["STATUS_MOVING"]

    LEF.steps(0, 10)
    status = LEF.get_statuses()
    assert status[0, 0] == constants["STATUS_MOVING"]
    assert status[0, 1] == constants["STATUS_PAUSED"]  # right leg of left LEF is paused
    assert status[1, 0] == constants["STATUS_PAUSED"]  # against left leg of right LEF
    assert status[1, 1] == constants["STATUS_MOVING"]

    LEF.steps(0, 100)
    status = LEF.get_statuses()
    assert status[0, 0] == constants["STATUS_PAUSED"]
    assert status[0, 1] == constants["STATUS_PAUSED"]
    assert status[1, 0] == constants["STATUS_PAUSED"]
    assert status[1, 1] == constants["STATUS_PAUSED"]

    # check that we are at the boundaries
    pos = LEF.get_LEFs()
    assert pos[0, 0] == 1
    assert pos[1, 1] == 98


def test_capture(): 
    """
    test if sites with 100% capture probability definitly capture LEFs
    Load one lef and place two capture sites around it, and confirm that it is captured 
    (and that statuses are correct)
    """

    N_LEFS = 1
    N = 100 

    load_array = np.zeros(N)
    load_array[40] = 1 

    unload_array = np.zeros((N, 3))  # no unloading

    capture_array = np.zeros((N, 2))  # no CTCF
    capture_array[37, 0] = 1
    capture_array[43, 1] = 1

    release_array = np.zeros(N) 

    pause_array = np.zeros(N)  # no pausing

    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True) 
    LEF.force_load_LEFs(np.array([[40, 41]]))
    LEF.steps(0, 5)
    status = LEF.get_statuses()
    assert status[0, 0] == constants["STATUS_CAPTURED"]
    assert status[0, 1] == constants["STATUS_CAPTURED"]
    pos = LEF.get_LEFs()
    assert pos[0, 0] == 37
    assert pos[0, 1] == 43




