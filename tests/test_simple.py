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

    unload_array = np.zeros((N, 4))  # no unloading

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
        assert pos[0, 0] == 34
        assert pos[0, 1] == 45 or pos[0, 1] == 46
        right_positions.append(pos[0, 1])
    assert len(set(right_positions)) == 2  # LEF was loaded at (39, 40) or at (39, 41)


def test_translocate_single():
    """
    Test if LEF translocates deterministically
    """

    N_LEFS = 1
    N = 100

    load_array = np.zeros(N)
    load_array[40] = 1

    unload_array = np.zeros((N, 4))  # no unloading

    capture_array = np.zeros((N, 2))  # no CTCF
    release_array = np.zeros(N)

    pause_array = np.zeros(N)  # no pausing

    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True)
    LEF.force_load_LEFs(np.array([[40, 41]]))
    LEF.steps(0, 5)
    pos = LEF.get_LEFs()
    assert pos[0, 0] == 35
    assert pos[0, 1] == 46


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

    unload_array = np.zeros((N, 4))  # no unloading

    capture_array = np.zeros((N, 2))  # no CTCF
    release_array = np.zeros(N)

    pause_array = np.zeros(N)  # no pausing

    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True)
    LEF.force_load_LEFs(np.array([[40, 41], [50, 51]]))
    LEF.steps(0, 10)
    pos = LEF.get_LEFs()
    assert pos[0, 0] == 30
    assert pos[0, 1] == 45
    assert pos[1, 0] == 46
    assert pos[1, 1] == 61


def test_occupied_array_matches():
    """
    Test if the occupied array matches the LEF positions in a LEFs with and without collision
    """
    N_LEFS = 3
    N = 100

    load_array = np.zeros(N)
    load_array[40] = 1
    load_array[41] = 1

    unload_array = np.zeros((N, 4))  # no unloading

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
            assert occ[pos[lef, leg]] == lef + N_LEFS * leg
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
    for i in range(1, N - 1):
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

    unload_array = np.zeros((N, 4))  # no unloading

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
    assert status[0, 1] == constants["STATUS_STALLED"]  # right leg of left LEF is stalled
    assert status[1, 0] == constants["STATUS_STALLED"]  # against left leg of right LEF
    assert status[1, 1] == constants["STATUS_MOVING"]

    LEF.steps(0, 100)
    status = LEF.get_statuses()
    assert status[0, 0] == constants["STATUS_STALLED"]
    assert status[0, 1] == constants["STATUS_STALLED"]
    assert status[1, 0] == constants["STATUS_STALLED"]
    assert status[1, 1] == constants["STATUS_STALLED"]

    # check that we are at the boundaries
    pos = LEF.get_LEFs()
    assert pos[0, 0] == 1
    assert pos[1, 1] == 98


def test_pausing():
    """
    Create a 100% pausing probability sites around a single LEF and confirm it is paused
    """
    N_LEFS = 1
    N = 100

    load_array = np.zeros(N)
    load_array[40] = 1

    unload_array = np.zeros((N, 4))  # no unloading

    capture_array = np.zeros((N, 2))  # no CTCF
    release_array = np.zeros(N)

    pause_array = np.zeros(N)  # no pausing
    pause_array[37] = 1
    pause_array[43] = 1

    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True)
    LEF.force_load_LEFs(np.array([[40, 41]]))
    LEF.steps(0, 5)
    status = LEF.get_statuses()
    assert status[0, 0] == constants["STATUS_PAUSED"]
    assert status[0, 1] == constants["STATUS_PAUSED"]
    pos = LEF.get_LEFs()
    assert pos[0, 0] == 37
    assert pos[0, 1] == 43


def test_capture():
    """
    test if sites with 100% capture probability definitly capture LEFs
    Load one lef and place two capture sites around it, and confirm that it is captured
    (and that statuses are correct)

    Confirm that capture works correctly: left leg is captured by left site, right leg by right site
    (left site is array 0, and right site is array 1)

    Confirm that capture doesn't work "the other way around" and the LEF moving left is not captured by the "right" site
    """
    N_LEFS = 1
    N = 100

    load_array = np.zeros(N)
    load_array[40] = 1

    unload_array = np.zeros((N, 4))  # no unloading

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

    # load again, but this time with the capture sites pointing "away" from the LEF
    capture_array = np.zeros((N, 2))  # no CTCF
    capture_array[37, 1] = 1
    capture_array[43, 0] = 1
    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True)
    LEF.force_load_LEFs(np.array([[40, 41]]))
    LEF.steps(0, 5)
    status = LEF.get_statuses()
    assert status[0, 0] == constants["STATUS_MOVING"]
    assert status[0, 1] == constants["STATUS_MOVING"]
    pos = LEF.get_LEFs()
    assert pos[0, 0] == 35
    assert pos[0, 1] == 46


def test_consistency_of_complex_system_and_non_overlapping_result():
    """
    Create a length-7000 system with 300 LEFs
    Create random values for all arrays:
    * random loading
    * random unloading on all 4 arrays, mean value of .02
    * random capture at 250 random sites in each direction, probability of .8
    * release of 0.02 at those sites
    * random pausing at 250 random sites, probability of .95

    Verify that the occupied array is consistent with the LEF positions after 10000 steps
    """
    N_LEFS = 300
    N = 7000

    load_array = np.random.random(N)

    unload_array = np.random.random((N, 4)) * 0.02
    capture_array = np.zeros((N, 2))
    release_array = np.zeros(N)
    pause_array = np.zeros(N)
    for i in range(250):
        site1 = np.random.randint(0, N)
        site2 = np.random.randint(0, N)
        capture_array[site1, 0] = 0.8
        capture_array[site2, 1] = 0.8
        release_array[site1] = 0.02
        release_array[site2] = 0.02
        site3 = np.random.randint(0, N)
        pause_array[site3] = 0.95

    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array)
    LEF.steps(0, 2000)
    pos = LEF.get_LEFs()
    occ = LEF.get_occupied()

    for lef in range(N_LEFS):
        for leg in range(2):
            assert occ[pos[lef, leg]] == lef + N_LEFS * leg
    # check it manually too (mostly for the person reading this to understand the logic)
    assert occ[0] == constants["OCCUPIED_BOUNDARY"]
    assert occ[-1] == constants["OCCUPIED_BOUNDARY"]
    all_pos = set(pos.flatten())
    # all other positions should be OCCIPUED_FREE
    for i in range(1, N - 1):
        if i not in all_pos:
            assert occ[i] == constants["OCCUPIED_FREE"]
    # we have so so many LEFs that we should have all statuses present
    statuses = LEF.get_statuses()
    status_set = set(statuses.flatten())
    assert constants["STATUS_MOVING"] in status_set
    assert constants["STATUS_PAUSED"] in status_set
    assert constants["STATUS_STALLED"] in status_set
    assert constants["STATUS_CAPTURED"] in status_set

    # verify that LEFs are non-overlapping

    counter_array = np.zeros(N)
    for i in range(N_LEFS):
        for j in range(2):
            counter_array[pos[i, j]] += 1
    assert np.all(counter_array <= 1)  # no position has more than one LEF


def test_watches():
    """
    Create a simple simulations.
    Then add watches for positions where a LEF will be in the future and test that watches are triggered correctly.
    """

    N_LEFS = 1
    N = 100
    load_array = np.zeros(N)
    load_array[40] = 1
    unload_array = np.zeros((N, 4))  # no unloading
    capture_array = np.zeros((N, 2))  # no CTCF
    release_array = np.zeros(N)
    pause_array = np.zeros(N)  # no pausing

    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True)
    LEF.force_load_LEFs(np.array([[40, 41]]))

    # add watches
    LEF.set_watches([35, 36, 37, 44, 45, 47], 100)
    LEF.steps_watch(0, 10)
    events = LEF.get_events()
    assert len(events) == 2  # third watch would not be activated
    assert events[0, 0] == 37  # the inner-most event was triggered first
    assert events[0, 1] == 44
    assert events[0, 2] == 2  # at the end of step 2 (step0 = 40->39 step2 = 38->37) - watch is triggered

    # You can reset watches and they reset. And get triggered again.
    LEF.set_watches([20, 61])  # will happen in 10 steps
    LEF.steps_watch(10, 15)
    events = LEF.get_events()
    assert len(events) == 0  # events reset correctly
    LEF.steps_watch(15, 20)
    events = LEF.get_events()
    assert len(events) == 1  # event happened after resetting watches

    N_LEFS = 2
    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=True)
    LEF.force_load_LEFs(np.array([[40, 41], [45, 46]]))
    LEF.set_watches(list(range(100)))  # can set many watches
    LEF.steps_watch(0, 100)
    events = LEF.get_events()
    assert len(events) == 200  # all watches happened - 2 LEFs for 100 steps
    events = LEF.get_events(reset=True)
    assert len(events) == 200  # still 200
    events = LEF.get_events()  # now it should be reset
    assert len(events) == 0  # all events are reset
