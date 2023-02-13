# -*- coding: utf-8 -*-
"""
Author   : alex
Created  : 2021-02-02 15:05:07
Modified : 2021-05-18 11:38:34

Comments : Correlation calculation functions using the boost-histogram package
           (http://boost-histogram.readthedocs.io/)

           Used in the mcp_analyze package
"""

# %% IMPORTS

# -- global

import numpy as np
import boost_histogram as bh
from numba import jit, njit
from tqdm import tqdm
from multiprocessing import Process, Queue
from scipy.spatial.distance import pdist


# %% FUNCTIONS

# - TOOLS


@njit()
def comp_diff(data):
    n_elements = len(data)
    n_diffs = int(0.5 * n_elements * (n_elements - 1))
    diffs = np.zeros(n_diffs)
    i = 0
    for j in range(n_elements):
        for k in range(j + 1, n_elements):
            diffs[i] = data[k] - data[j]
            i += 1
    return diffs


@njit()
def comp_sum(data):
    n_elements = len(data)
    n_diffs = int(0.5 * n_elements * (n_elements - 1))
    diffs = np.zeros(n_diffs)
    i = 0
    for j in range(n_elements):
        for k in range(j + 1, n_elements):
            diffs[i] = data[k] + data[j]  # here is the diff with comp_diff
            i += 1
    return diffs


@jit(nopython=True, nogil=True)
def comp_diff_2arrays(x, y):
    n_x = len(x)
    n_y = len(y)
    n_diffs = int(n_x * n_y)
    diffs = np.zeros(n_diffs)
    i = 0
    for j in range(n_x):
        for k in range(n_y):
            diffs[i] = y[k] - x[j]
            i += 1
    return diffs


def comp_diff_pdist(data):
    n_elements = len(data)
    diffs = pdist(np.reshape(data, (n_elements, 1)))
    return diffs


# - CORRELATION FUNCTIONS


def _bunch_corr(G2, kx, ky, kz, dk_range, diff=comp_diff):
    """
    correlation computation subroutine, taking the whole run at once
    """
    # compute diff
    dkx = diff(kx)
    dky = diff(ky)
    dkz = diff(kz)

    # post-filter
    mask_x = (dkx > -dk_range) * (dkx < dk_range)
    mask_y = (dky > -dk_range) * (dky < dk_range)
    mask_z = (dkz > -dk_range) * (dkz < dk_range)
    mask = mask_x * mask_y * mask_z

    dkx = dkx[mask]
    dky = dky[mask]
    dkz = dkz[mask]

    # fill histogram
    G2.fill(dkx, dky, dkz)
    G2.fill(-dkx, -dky, -dkz)

    return G2


def _indiv_corr(G2, kx, ky, kz, dk_range, atom_list=None):
    """
    correlation computation subroutine, one atom at the time
    """
    if atom_list is None:
        atom_list = range(len(kx))

    for i_atom in atom_list:
        # compute diff
        dkx = kx[i_atom] - kx
        dky = ky[i_atom] - ky
        dkz = kz[i_atom] - kz

        # post-filter
        mask_x = (dkx > -dk_range) * (dkx < dk_range)
        mask_y = (dky > -dk_range) * (dky < dk_range)
        mask_z = (dkz > -dk_range) * (dkz < dk_range)
        mask = mask_x * mask_y * mask_z
        mask[i_atom] = False

        dkx = dkx[mask]
        dky = dky[mask]
        dkz = dkz[mask]

        # fill histogram
        G2.fill(dkx, dky, dkz)
        G2.fill(-dkx, -dky, -dkz)

    return G2


def _indiv_corr_multiproc(G2, kx, ky, kz, dk_range, n_points, n_proc=4):
    """
    correlation computation subroutine, one atom at the time
    """
    # split atoms
    all_atoms = range(len(kx))
    atom_list_split = np.array_split(all_atoms, n_proc)

    # define process function
    def corr_subprocess(queue, atom_list):
        # prepare G2
        hist_grid = bh.axis.Regular(
            bins=n_points, start=-dk_range, stop=dk_range
        )
        G2 = bh.Histogram(
            hist_grid, hist_grid, hist_grid, storage=bh.storage.Unlimited()
        )
        # compute
        G2 = _indiv_corr(G2, kx, ky, kz, dk_range, atom_list)
        # store
        queue.put(G2)

    # prepare queue and processes
    queue = Queue()
    processes = []
    for atom_list in atom_list_split:
        p = Process(target=corr_subprocess, args=(queue, atom_list))
        processes.append(p)

    # start
    for p in processes:
        p.start()

    # harvest results
    for p in processes:
        G2_part = queue.get()
        G2 = G2 + G2_part

    # wait for jobs to finish
    for p in processes:
        p.join()

    return G2


def correlate(
    data,
    dk_range=0.3,
    n_points=200,
    use_tqdm=False,
    diff_method="default",
    n_proc_indiv=4,
):
    """
    Computes the (not normalized) G2 function for the loaded data
    Lowest level function

    Parameters
    ----------
    data : pandas dataframe
        momentum collection
    dk_range : float, optional
        range for the impulsion grid
    n_points : int, optional
        number of points for the impulsion grid
    """

    # - choose diff method
    # dictionnary of implemented methods
    diff_method_dict = {
        "default": comp_diff,
        "pdist": comp_diff_pdist,
        "indiv_multiproc": None,
        "indiv": None,
        "sum": comp_sum,  # for k/-k correlations
    }
    # check that the method is implemented
    errmsg = "'%s' diff method not implemented. " % diff_method
    errmsg += "Available methods : " + ", ".join(diff_method_dict.keys())
    assert diff_method in diff_method_dict, errmsg
    # pick requested method
    diff = diff_method_dict[diff_method]

    # - initialize G2 histogram
    # grid (same for all directions)
    hist_grid = bh.axis.Regular(bins=n_points, start=-dk_range, stop=dk_range)
    # create empty histogram
    G2 = bh.Histogram(
        hist_grid, hist_grid, hist_grid, storage=bh.storage.Unlimited()
    )
    # prepare atom number list
    n_atoms = np.zeros_like(data.index)

    # - loop on data sets
    # prepare tqdm if requested
    if use_tqdm:
        index_list = tqdm(data.index)
    else:
        index_list = data.index
    # loop
    for i_processed, i_set in enumerate(index_list):
        # get run data
        kx = 1.0 * data.kx[i_set]
        ky = 1.0 * data.ky[i_set]
        kz = 1.0 * data.kz[i_set]

        # save atoms
        n_atoms[i_processed] = len(kx)

        # compute correlation
        if diff_method == "indiv":
            G2 = _indiv_corr(G2, kx, ky, kz, dk_range)
        elif diff_method == "indiv_multiproc":
            G2 = _indiv_corr_multiproc(
                G2, kx, ky, kz, dk_range, n_points, n_proc_indiv,
            )
        else:
            G2 = _bunch_corr(G2, kx, ky, kz, dk_range, diff=diff)

    return G2, n_atoms


def compute_correlation(
    data, n_runs_max=None, multi_processing=False, n_proc=1, **correlate_params
):
    """
    Wrapper for the correlate() function. Filters the data, and dispatch in
    several processes if 'multi_processing' is set to true

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the atoms' momentum
    n_runs_max : None, optional
        when an integer is given, sets the maximum number of run considerer
        for the g2 calculation (otherwise the whole dataset is used)
    multi_processing : bool, optional
        enables multi processing for the correlation computation
    n_proc : int, optional
        number of processes to use in the multi processing mode
    **correlate_params
        other parameters sent to correlate() (see documentation)
    """

    # - prepare datasets
    # get number of sets and set upper limit (if requested)
    if n_runs_max is not None:
        n_sets = len(data)
        n_sets = np.min([n_sets, n_runs_max])
        indexes = data.index[0:n_sets]
        data_filtered = data.filter(items=indexes, axis=0)
    else:
        data_filtered = data

    # - compute correlation
    # simple case : single process
    if not multi_processing:
        G2, n_atoms = correlate(data_filtered, **correlate_params)

    # multi processing case
    else:
        # handle tqdm in the multiprocessing case
        if "use_tqdm" in correlate_params:
            use_tqdm = correlate_params.pop("use_tqdm")
        else:
            use_tqdm = False

        # split data sets
        n_sets = len(data_filtered)
        sets_list = np.array_split(np.arange(n_sets), n_proc)

        # define process function
        def comp_g2_subprocess(queue, data, use_tqdm):
            G2, n_atoms = correlate(
                data, use_tqdm=use_tqdm, **correlate_params
            )
            queue.put((G2, n_atoms))

        # prepare queue and processes
        queue = Queue()
        processes = []
        for sets in sets_list:
            sub_data_set = data_filtered.filter(items=sets, axis=0)
            args = (queue, sub_data_set, use_tqdm)
            p = Process(target=comp_g2_subprocess, args=args)
            processes.append(p)
            use_tqdm = False  # so that only the first one is set to True

        # start
        for p in processes:
            p.start()

        # harvest results
        G2 = 0
        n_atoms = np.array([])
        for p in processes:
            G2_part, n_atoms_part = queue.get()
            G2 = G2 + G2_part
            n_atoms = np.concatenate((n_atoms, n_atoms_part))
        # wait for jobs to finish
        for p in processes:
            p.join()

    # - prepare output
    correlate_params["multi_processing"] = multi_processing
    correlate_params["n_runs_max"] = n_runs_max
    results = {"hist": G2, "parameters": correlate_params, "n_atoms": n_atoms}

    return results
