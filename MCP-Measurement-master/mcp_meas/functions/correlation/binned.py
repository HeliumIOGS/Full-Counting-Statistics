# -*- coding: utf-8 -*-
"""
Author   : alex
Created  : 2021-02-09 11:35:58
Modified : 2021-05-18 14:10:55

Comments : Correlation calculation functions uning histogramming (binning)
"""

# %% IMPORTS

# -- global

import numpy as np
import boost_histogram as bh
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial


# %% FUNCTIONS

# == low-level correlation calculation functions


def GN_0(
    k,
    k_range,
    dk_bin_list,
    n,
    remove=True,
    shift=(0, 0, 0),
    expand_range_if_needed=False,
    count_events=False,
):
    """
    Local G^(n) correlation calculation
    """
    # -- prepare real bins
    if expand_range_if_needed:
        n_bins_list = np.uint(np.ceil(2 * k_range / dk_bin_list))
        k_range_list = 0.5 * n_bins_list * dk_bin_list
    else:
        # compute number of bins, and remove doublons
        n_bins_list = np.uint(2 * k_range / dk_bin_list)
        n_bins_list = np.unique(n_bins_list)
        k_range_list = [k_range for i in range(len(n_bins_list))]

    # -- initialize G0
    G0 = np.zeros_like(n_bins_list, dtype=float)  # float to avoid overflow !!
    bins = np.zeros_like(n_bins_list, dtype=float)
    kx = k[0] + shift[0]
    ky = k[1] + shift[1]
    kz = k[2] + shift[2]
    if count_events:
        N_events = np.zeros_like(n_bins_list, dtype=float)
    # -- loop on all dk_bins dk_bin_list
    for i_bin, n_bins in enumerate(n_bins_list):
        # - generate histogram
        # prepare grid
        k_range = k_range_list[i_bin]
        grid = bh.axis.Regular(bins=n_bins, start=-k_range, stop=k_range)
        # create histogram
        density = bh.Histogram(
            grid, grid, grid, storage=bh.storage.Unlimited()
        )
        centers = density.axes[0].centers
        # fill with data
        density.fill(kx, ky, kz)
        # convert to array
        density = density.view()
        # save total atom number
        N = np.sum(density)

        # -- compute G^(n)(0)
        # remove useless points (will lead to zero in calculation)
        if remove:
            density = density[density > n - 1]
        # count events if requested
        if count_events:
            N_events[i_bin] = len(density[density > n - 1])
        # perform multiplication
        # x = rho * (rho-1) * ... * (rho - n + 1)
        x = 1
        for step in range(n):
            x = x * density
            density -= 1
        # store value
        G0[i_bin] = np.sum(x)
        bins[i_bin] = centers[1] - centers[0]

    # -- return
    if count_events:
        return G0, bins, N, N_events
    else:
        return G0, bins, N


def GN_0_anisotropic(
    k,
    k_range,
    dk_bin_list,
    dk_bin_trans,
    n,
    remove=True,
    axis="x",
    expand_range_if_needed=False,
    count_events=False,
):
    """
    Local G^(n) correlation calculation
    """
    # -- prepare real bins
    if expand_range_if_needed:
        n_bins_list = np.uint(np.ceil(2 * k_range / dk_bin_list))
        k_range_list = 0.5 * n_bins_list * dk_bin_list
    else:
        # compute number of bins, and remove doublons
        n_bins_list = np.uint(2 * k_range / dk_bin_list)
        n_bins_list = np.unique(n_bins_list)
        k_range_list = [k_range for i in range(len(n_bins_list))]

    # -- initialize G0
    G0 = np.zeros_like(n_bins_list, dtype=float)  # float to avoid overflow !!

    bins = np.zeros_like(n_bins_list, dtype=float)
    n_bins_trans = int(2 * k_range / dk_bin_trans)
    grid_trans = bh.axis.Regular(
        bins=n_bins_trans, start=-k_range, stop=k_range
    )
    kx = k[0]
    ky = k[1]
    kz = k[2]
    if count_events:
        N_events = np.zeros_like(n_bins_list, dtype=float)
    # -- loop on all dk_bins dk_bin_list
    for i_bin, n_bins in enumerate(n_bins_list):
        # - generate histogram
        # prepare grid
        k_range = k_range_list[i_bin]
        grid = bh.axis.Regular(bins=n_bins, start=-k_range, stop=k_range)
        # create histogram
        if axis == "x":
            density = bh.Histogram(
                grid, grid_trans, grid_trans, storage=bh.storage.Unlimited()
            )
            centers = density.axes[0].centers
        elif axis == "y":
            density = bh.Histogram(
                grid_trans, grid, grid_trans, storage=bh.storage.Unlimited()
            )
            centers = density.axes[1].centers
        elif axis == "z":
            density = bh.Histogram(
                grid_trans, grid_trans, grid, storage=bh.storage.Unlimited()
            )
            centers = density.axes[2].centers

        # fill with data
        density.fill(kx, ky, kz)
        # convert to array
        density = density.view()
        # save total atom number
        N = np.sum(density)

        # -- compute G^(n)(0)
        # remove useless points (will lead to zero in calculation)
        if remove:
            density = density[density > n - 1]
        # count events if requested
        if count_events:
            N_events[i_bin] = len(density[density > n - 1])
        # perform multiplication
        # x = rho * (rho-1) * ... * (rho - n + 1)
        x = 1
        for step in range(n):
            x = x * density
            density -= 1
        # store value
        G0[i_bin] = np.sum(x)
        bins[i_bin] = centers[1] - centers[0]

    # -- return
    if count_events:
        return G0, bins, N, N_events
    else:
        return G0, bins, N


# == batch processing functions


def batch_process(data_list, Gfunc, n_proc=4, **Gfunc_params):
    """
    Parallel processing of data_list
    data_list must be a list :
        data_list = [(kx_1, ky_1, kz_1), ... , (kx_N, ky_N, kz_N)]
    where kx_1 is a list (array-like) of momentum, corresponding to run #1

    Gfunc must output : G0, bins, N_tot, with G0 and bins 1D arrays of same
    length, and N_tot a scalar

    Gfunc params must contain the parameter 'n', corresponding to the order
    of the correlation function
    """

    # define a local version of the correlation function, with the good params
    # def Gpool(k):
    #     return GN_0(k[0], k[1], k[2], **Gfunc_params)

    Gpool = partial(Gfunc, **Gfunc_params)
    # compute using a pool
    with Pool(n_proc) as pool:
        out = []
        for i in tqdm(pool.imap(Gpool, data_list), total=len(data_list)):
            out.append(i)

    # combine outputs in a handy format
    n_bins = len(out[0][0])
    n_runs = len(out)
    G0 = np.zeros((n_bins, n_runs))
    N = np.empty_like(G0)
    for i_run in range(n_runs):
        G0[:, i_run] = out[i_run][0]
        N[:, i_run] = out[i_run][2]
    bins = out[0][1]

    # store
    results = {"G0": G0, "N": N, "bins": bins, "out": out}

    # compute some quantities, summing or averaging over all runs
    n = Gfunc_params["n"]
    norm = np.sum(N[0, :] ** n)
    results["g0_sum"] = np.sum(G0, 1) / norm
    if n_runs > 1:
        results["g0_mean"] = np.mean(G0 / N ** n, 1)
        results["g0_std"] = np.std(G0 / N ** n, 1)
        results["g0_err"] = results["g0_std"] / np.sqrt(n_runs)

    return results
