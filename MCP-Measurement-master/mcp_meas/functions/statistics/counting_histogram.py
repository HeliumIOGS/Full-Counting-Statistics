# -*- coding: utf-8 -*-
"""
Author   : alex
Created  : 2021-05-18 14:13:58
Modified : 2021-06-03 15:26:40

Comments : Local counting statistics calculation
"""

# %% IMPORTS

# -- global
import gc
import numpy as np
import boost_histogram as bh
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

# %% FUNCTIONS

# == low-level histogramming function

def count_hist_old(
    k, k_range, dk_bin_list, remove=True, shift=(0, 0, 0), expand_range_if_needed=False,
):
    """
    Local counting histogram, takes a list of bin sizes as input
    (dk_bin_list)
    """
    # -- prepare real bins
    if expand_range_if_needed:
        n_bins_list = np.uint(np.ceil(2 * k_range / dk_bin_list))
        k_range_list = 0.5 * n_bins_list * dk_bin_list
    else:
        # compute number of bins, and remove doublons
        n_bins_list = np.uint(k_range / dk_bin_list)
        n_bins_list = np.unique(n_bins_list)
        k_range_list = [k_range for i in range(len(n_bins_list))]
    # -- initialize G0
    hist = []
    bins = np.zeros_like(n_bins_list, dtype=float)

    kx = np.squeeze(k[0]) + shift[0]
    ky = np.squeeze(k[1]) + shift[1]
    kz = np.squeeze(k[2]) + shift[2]
    # -- loop on all dk_bins dk_bin_list
    for i_bin, n_bins in enumerate(n_bins_list):
        # - generate histogram
        # prepare grid
        k_range = k_range_list[i_bin]
        grid = bh.axis.Regular(bins=n_bins, start=-k_range, stop=k_range)
        # create histogram
        density = bh.Histogram(grid, grid, grid, storage=bh.storage.Unlimited())
        centers = density.axes[0].centers
        # fill with data
        density.fill(kx, ky, kz)
        # convert to array
        density = density.view()
        # store value
        hist.append(density)
        bins[i_bin] = dk_bin_list
        #bins[i_bin] = centers[1] - centers[0]
    
    N = np.sum(density)
    # -- return
    return hist, centers, bins, N, n_bins_list


def count_hist(
    k, k_range, n_bin, shift=(0, 0, 0),
):
    """
    Local counting histogram, takes a single
    """
    # -- prepare k_range and n_bin
    # convert to array
    k_range = np.squeeze(np.asarray(k_range))
    n_bin = np.squeeze(np.asarray(n_bin))
    # k_range should have a shape of either 0 or 3
    msg = "'{}' shoud be either a number or a list of 3 numbers"
    for x, x_str in zip([n_bin, k_range], ["n_bin", "k_range"]):
        assert x.shape in [(), (3,)], msg.format(x_str)
    # convert to list of 3
    if k_range.shape == ():
        k_range = (k_range, k_range, k_range)
    if n_bin.shape == ():
        n_bin = (n_bin, n_bin, n_bin)

    # -- shift
    kx = np.squeeze(k[0]) + shift[0]
    ky = np.squeeze(k[1]) + shift[1]
    kz = np.squeeze(k[2]) + shift[2]

    # -- generate histogram
    # prepare grid
    grid = []
    for kr, nb in zip(k_range, n_bin):
        grid.append(bh.axis.Regular(bins=nb, start=-kr, stop=kr))
    # create histogram
    density = bh.Histogram(*grid, storage=bh.storage.Unlimited())
    # compute dk
    dk = [ax.centers[1] - ax.centers[0] for ax in density.axes]
    # fill with data
    density.fill(kx, ky, kz)
    # convert to array
    hist = density.view()

    # -- del
    del density, grid

    # -- return
    return hist, dk


def count_hist_non_uniform(
    k, gridX, gridY, gridZ, remove=True, shift=(0, 0, 0), expand_range_if_needed=False,
):
    """
    Local counting histogram
    """
    # -- initialize G0
    hist = []

    kx = np.squeeze(k[0]) + shift[0]
    ky = np.squeeze(k[1]) + shift[1]
    kz = np.squeeze(k[2]) + shift[2]
    # -- loop on all dk_bins dk_bin_list
    #for i_bin, n_bins in enumerate(n_bins_list):
    # - generate histogram
    # create histogram
    density = bh.Histogram(gridX, gridY, gridZ, storage=bh.storage.Unlimited())
    centers = (density.axes[0].centers, density.axes[1].centers, density.axes[2].centers)
    bins = (density.axes[0].widths[0], density.axes[1].widths[0], density.axes[2].widths[0])
    # fill with data
    density.fill(kx, ky, kz)
    # convert to array
    density = density.view()
    # store value
    hist.append(density)
    
    
    N = np.sum(density)
    # -- return
    return hist, centers, bins, N

# == batch processing functions


def batch_process_count_hist_old(data_list, n_proc=4, **hist_params):
    """
    Parallel processing of data_list
    data_list must be a list :
        data_list = [(kx_1, ky_1, kz_1), ... , (kx_N, ky_N, kz_N)]
    where kx_1 is a list (array-like) of momentum, corresponding to run #1

    returns
    """

    # define a local version of the histogram function with the good parameters
    histPool = partial(count_hist_old, **hist_params)
    # compute using a pool
    with Pool(n_proc) as pool:
        out = []
        for i in tqdm(pool.imap(histPool, data_list), total=len(data_list)):
            out.append(i)

    return out


def batch_process_count_hist(data_list, n_proc=4, **hist_params):
    """
    Parallel processing of data_list
    data_list must be a list :
        data_list = [(kx_1, ky_1, kz_1), ... , (kx_N, ky_N, kz_N)]
    where kx_1 is a list (array-like) of momentum, corresponding to run #1
    """

    # define a local version of the histogram function with the good parameters
    histPool = partial(count_hist, **hist_params)
    # compute using a pool
    with Pool(n_proc) as pool:
        out = []
        for i in tqdm(pool.imap(histPool, data_list), total=len(data_list)):
            out.append(i)
    return out


def batch_process_count_hist_non_uniform(data_list, n_proc=4, **hist_params):
    """
    Parallel processing of data_list
    data_list must be a list :
        data_list = [(kx_1, ky_1, kz_1), ... , (kx_N, ky_N, kz_N)]
    where kx_1 is a list (array-like) of momentum, corresponding to run #1

    returns
    """

    # define a local version of the histogram function with the good parameters
    histPool = partial(count_hist_non_uniform, **hist_params)
    # compute using a pool
    with Pool(n_proc) as pool:
        out = []
        for i in tqdm(pool.imap(histPool, data_list), total=len(data_list)):
            out.append(i)

    return out


# == high-level functions
def analyze_dataset(data_list, n_proc=4, **hist_params):
    """
    calls the parallel processing function `batch_process_count_hist`, and
    process the output to return it in a more user-friendly format
    """
    # -- call the batch processing function
    out = batch_process_count_hist(data_list, n_proc=n_proc, **hist_params)

    # -- process the output
    # - momentum spacing
    dk = out[0][1]
    # - gather all histograms into one array
    n_runs = len(out)
    nx, ny, nz = out[0][0].shape
    hist = np.zeros((n_runs, nx, ny, nz))
    for i_run, hist_run in enumerate(out):
        hist[i_run, :, :, :] = hist_run[0]

    # -- delete and collect
    del out
    gc.collect()

    # -- return
    res = {"dk": dk, "hist": hist, "params": hist_params}

    return res
