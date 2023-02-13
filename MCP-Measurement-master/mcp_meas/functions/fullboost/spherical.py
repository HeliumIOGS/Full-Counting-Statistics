# -*- coding: utf-8 -*-
"""
Author   : Alexandre
Created  : 2021-06-17 09:15:24
Modified : 2021-06-21 11:32:49

Comments : using boost-histogram in spherical coordinates
"""

# %% IMPORTS

# -- global
import boost_histogram as bh
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple

# -- local
from . import spherical_slicing as ss
from ..statistics import distributions as dist


# %% HISTOGRAMMING FUNCTIONS
"""Functions meant to compute density histograms from raw exp. data"""


def slice_histogram(kr, kth, kph, run=None, start=0, stop=0.1, n_slice=10):
    """Computes an histogram in spherical coordinates, shell by shell, trying to divide
    space in order to have almost isovolumic boxes. If `run` is given, the histogram
    is computed run by run ; otherwise, only the density is computed"""

    # -- make sure phi and rho are in the good quadrant
    kth = np.mod(kth, np.pi)
    kph = np.mod(kph, 2 * np.pi)

    # -- prepare rho grid
    re = np.linspace(start, stop, n_slice + 1)  # edges
    rc = re[:-1] + 0.5 * np.diff(re)  # centers

    # -- compute
    hist_list = []
    volume_list = []
    for i in tqdm(range(n_slice), desc="Histogramming"):
        # get the good theta and phi grid
        slicing = ss.shell_slicing_3D(re[i], re[i + 1])
        # compute the volumes, and store them
        volumes = ss.real_theta_volumes(
            slicing["theta_grid"], slicing["phi_grid"], re[i], re[i + 1]
        )
        volume_list.append(volumes)
        # prepare spatial histograms grids
        phi_grid = bh.axis.Regular(bins=slicing["Nphi"], start=0, stop=2 * np.pi)
        theta_grid = bh.axis.Variable(slicing["theta_grid"])
        r_grid = bh.axis.Regular(bins=1, start=re[i], stop=re[i + 1])
        grids = [r_grid, theta_grid, phi_grid]
        data = [kr, kth, kph]
        # add a run grid, if needed
        if run is not None:
            n_runs = int(np.max(run))
            run_grid = bh.axis.Integer(start=0, stop=n_runs + 1)
            grids.insert(0, run_grid)
            data.insert(0, run)
        # init histogram, fill it and store it
        hist = bh.Histogram(*grids)
        hist.fill(*data)
        hist_list.append(hist)

    # -- return result
    res = {"h": hist_list, "volumes": volume_list, "rc": rc, "re": re}
    return res


# %% HISTOGRAM ANALYSIS : DENSITY
"""Functions to analyse the histograms (or the raw data) to get density information"""


def extract_radial_density(res, plot=True):
    """Extracts mean radial density for an histogram (computed by `slice_histogram`)"""
    # -- process
    # prepare h list
    h = res["h"][0]
    if len(h.shape) == 4:
        # then the first dimension is the run number
        h_list = [h[::sum, :, :, :] for h in res["h"]]
    else:
        h_list = res["h"]
    # get n_mean & n_std slice by slice
    n_mean = np.array([np.mean(h) for h in h_list])
    n_std = np.array([np.std(h) for h in h_list])
    # get slices radial centers
    rc = res["rc"]
    # get number of boxes
    n_box = np.array([h.size for h in h_list])
    # get volumes
    v_mean = np.array([np.mean(v) for v in res["volumes"]])
    v_std = np.array([np.std(v) for v in res["volumes"]])
    # normalize
    norm = v_mean / np.mean(v_mean)
    # output
    x = namedtuple("x", ["value", "error"])
    res = namedtuple("result", ["n_at", "rc", "vol", "n_box"])
    out = res(n_at=x(n_mean, n_std), rc=rc, vol=x(v_mean, v_std), n_box=n_box)
    # -- plot ?
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        p = dict(fmt=":o", markersize=5, linewidth=1)
        # atom number
        ax[0].errorbar(rc, n_mean, yerr=n_std, label="raw", **p)
        ax[0].errorbar(rc, n_mean / norm, yerr=n_std / norm, label="rect", **p)
        ax[0].legend()
        ax[0].set_ylabel("atom per box")
        # boxes volumes
        ax[1].errorbar(rc, v_mean / np.mean(v_mean), yerr=v_std / np.mean(v_mean), **p)
        ax[1].set_ylabel("relative boxes volumes")
        for cax in ax:
            cax.grid()
            cax.set_xlabel("kr")
        plt.show()
    return out


def compute_radial_density(
    kr, start=0, stop=0.1, n_slice=10, use_tqdm=True, norm_to_center=True
):
    """computes the radial density directly from the raw experimental data"""
    # -- prepare rho grid
    re = np.linspace(start, stop, n_slice + 1)  # edges
    rc = re[:-1] + 0.5 * np.diff(re)  # centers

    # -- compute
    density = np.zeros_like(rc)
    it = tqdm(range(n_slice), desc="Density") if use_tqdm else range(n_slice)
    for i in it:
        # get limits
        r0 = re[i]
        r1 = re[i + 1]
        # get atom number
        i_mask = (kr > r0) * (kr < r1)
        n_atoms = np.sum(i_mask)
        # normalize
        v_slice = (4 / 3) * np.pi * (r1 ** 3 - r0 ** 3)
        density[i] = n_atoms / v_slice

    # norm
    if norm_to_center:
        density /= density[0]

    return rc, density


# %% HISTOGRAM ANALYSIS : COUNTING STATISTICS
"""Compute counting statistics from a given histogram"""


def extract_radial_stats(res, use_tdqm=True, compute_th_distrib=True):
    """Extracts the counting statistics slice by slice"""
    # -- check inputs
    # check that the "run" axis is present
    h = res["h"][0]
    if len(h.shape) != 4:
        return
    # get slices radial centers
    rc = res["rc"]
    n_slices = len(rc)
    it = tqdm(range(n_slices), desc="Stats") if use_tdqm else range(n_slices)

    # -- process slice by slice
    # init arrays
    count_hist = []
    n_mean = np.zeros_like(rc)
    n_std = np.zeros_like(rc)
    n_box = np.zeros_like(rc)
    x_mean = np.zeros_like(rc)  # x = (<x^2> - <x>^2) / <x>
    x_std = np.zeros_like(rc)
    poisson = []
    thermal = []

    # loop on slices
    for i in it:
        # get histogram
        h = res["h"][i]
        # get number of boxes
        n_box[i] = np.prod(h.shape[1:])
        # get atom number and std
        n_mean[i] = np.mean(h)
        n_std[i] = np.std(h)
        # get "x" factor = variance / mean
        m = np.mean(h, axis=0)
        s = np.std(h, axis=0)
        m[m == 0] = np.nan
        x = s ** 2 / m
        x_mean[i] = np.nanmean(x)
        x_std[i] = np.nanstd(x)

        # compute full distrib
        count_grid = bh.axis.Integer(0, 1, growth=True)
        c_hist = bh.Histogram(count_grid)
        c_hist.fill(np.ravel(h))
        count_hist.append(c_hist)

        # compute poisson and thermal
        if compute_th_distrib:
            n_mean_list = np.ravel(np.mean(h, axis=0))
            n_counts = c_hist.axes[0].edges[:-1]
            n_list = np.linspace(0, np.max(n_counts), 200)
            h_poisson = np.zeros_like(n_list, dtype=float)
            h_thermal = np.zeros_like(n_list, dtype=float)
            for i_n, n in enumerate(n_mean_list):
                h_poisson += dist.poisson(n_list, n)
                h_thermal += dist.thermal(n_list, n)
            h_thermal /= len(n_mean_list)
            h_poisson /= len(n_mean_list)
            thermal.append([n_list, h_thermal])
            poisson.append([n_list, h_poisson])

    # also get volumes
    v_mean = np.array([np.mean(v) for v in res["volumes"]])
    v_std = np.array([np.std(v) for v in res["volumes"]])
    # output
    x = namedtuple("x", ["value", "error"])
    o = namedtuple(
        "result", ["rc", "vol", "n_at", "n_box", "x", "hist", "poisson", "thermal"]
    )
    out = o(
        rc=rc,
        vol=x(v_mean, v_std),
        n_at=x(n_mean, n_std),
        n_box=n_box,
        x=x(x_mean, x_std),
        hist=count_hist,
        poisson=poisson,
        thermal=thermal,
    )

    return out


# %% HISTOGRAM ANALYSIS : CORRELATIONS
"""Compute correlations from a given histogram"""


def extract_radial_gn(res, n=2, use_tdqm=True):
    """Extracts the local g^(n) slice by slice"""
    # -- check inputs
    # check that the "run" axis is present
    h = res["h"][0]
    if len(h.shape) != 4:
        return
    # get slices radial centers
    rc = res["rc"]
    n_slices = len(rc)
    it = tqdm(range(n_slices), desc="Correlations") if use_tdqm else range(n_slices)

    # -- process slice by slice
    # init arrays
    n_list = np.arange(1, n + 1)
    gn_mean = {i: np.zeros_like(rc) for i in n_list}
    gn_std = {i: np.zeros_like(rc) for i in n_list}
    n_box = np.zeros_like(rc)
    # loop on slices
    for i_slice in it:
        # get histogram
        h = res["h"][i_slice]
        # get number of boxes
        n_box[i_slice] = h.size
        # - compute gn
        # numerator
        Gn = {}
        GnSq = {}
        x = h.copy()
        prod = 1
        prodSq=1
        n_run = h.shape[0]
        for i in n_list:
            prod = prod * x
            prodSq = prod * prod
            x = x + (-1)
            Gn[i] = prod[::sum, :, :, :] / n_run
            GnSq[i] = prodSq[::sum, :, :, :] / n_run
            
        # normalize
        norm = 1
        for i in n_list:
            norm = norm * Gn[1]
            gn = Gn[i] / norm
            gn_mean[i][i_slice] = np.nanmean(gn)
             
            gn_std[i][i_slice] = np.nanstd(gn) 
            # Modified by David on 20/10/21 to include a value for the central slice associated 
            #    to the inverse of (number of pairs)^1/2 or (number of triplets)^1/3 or ....:
            #gn_std[i][i_slice] = np.maximum(1/(np.nanmean(Gn[i]))**(1/i),np.nanstd(gn))
            # Modified by David on 05/05/22 to include a value for the central slice associated 
            #gn_std[i][i_slice] = np.sqrt(GnSq[i] + (-1)*Gn[i]*Gn[i])/norm /np.sqrt(n_run)
            #print(np.nanstd(gn))

    # output
    x = namedtuple("x", ["value", "error"])
    o = namedtuple("result", ["rc", "gn", "n_box", "n_max"])
    out = o(
        rc=rc,
        gn=x(gn_mean, gn_std),
        n_box=n_box,
        n_max=n,
    )

    return out
