# -*- coding: utf-8 -*-
"""
Author   : Alexandre
Created  : 2021-06-14 14:45:20

Comments : a collection of functions taking full advantage of the boost-histogram
           library
"""

# %% IMPORTS
import time
import boost_histogram as bh
import numpy as np
from tqdm import tqdm


# %% HISTOGRAMMING FUNCTIONS
"""Functions meant to compute density histograms from raw exp. data"""


def hist(
    run,
    kx,
    ky,
    kz,
    k_range=0.1,
    dk=0.02,
    n_bin=None,
    ensure_odd=True,
    shift=(0, 0, 0),
    compute_random=True,
    verbose=True,
    random_version=1,
):
    # -- verbose ?
    def _p(x):
        if verbose:
            print(x)

    # -- parsing inputs
    if n_bin is None:
        n_bin = int(2 * k_range / dk)
        if ensure_odd:
            n_bin += (n_bin % 2) - 1

    params = dict(k_range=k_range, n_bin=n_bin, shift=shift)

    # -- prepare output
    out = {
        "p": params,
        "h": None,
        "r": None,
        "d": None,
        "help": "c = hist corr ; r = hist rand ; d = density ; p = parameters",
    }

    # -- prepare grids
    n_run = int(np.max(run))
    k_grid = bh.axis.Regular(bins=n_bin, start=-k_range, stop=k_range)
    k_grid_list = [k_grid, k_grid, k_grid]
    run_grid = bh.axis.Integer(0, n_run)
    data_list = [1 * kx + shift[0], 1 * ky + shift[1], 1 * kz + shift[2]]

    # -- normal histogram
    _p("+ compute histogram")
    _p(f"    > range {k_range}")
    _p(f"    > bins {n_bin}")
    t0 = time.time()
    # init
    hist = bh.Histogram(run_grid, *k_grid_list)
    hist.reset()
    # fill
    hist.fill(run, *data_list)
    # store
    out["c"] = hist
    tf = time.time()
    _p(f"FINISHED IN {tf-t0:.2f} s \n")

    # -- density
    _p("+ compute density")
    t0 = time.time()
    # init
    density = bh.Histogram(*k_grid_list)
    # fill
    density.fill(*data_list)
    # store
    out["d"] = density
    tf = time.time()
    _p(f"FINISHED IN {tf-t0:.2f} s \n")

    # -- random
    if compute_random:
        _p("+ compute random")
        t0 = time.time()
        # shuffle runs
        if random_version == 1:
            run_rand = np.random.randint(0, np.max(run) + 1, len(run))
        else:
            run_rand = 1.0 * run
            np.random.shuffle(run_rand)
        # init
        hist_rand = bh.Histogram(run_grid, *k_grid_list)
        hist_rand.reset()
        # fill
        hist_rand.fill(run_rand, *data_list)
        # store
        out["r"] = hist_rand
        tf = time.time()
        _p(f"FINISHED IN {tf-t0:.2f} s \n")

    return out


# %% CORRELATION FUNCTIONS
"""Functions to compute correlations g^(n) from the run-by-run density histograms"""


def _Gn(h, n):
    x = h.copy()
    prod = 1
    for i in range(n):
        prod = prod * x
        x = x + (-1)

    n_run = h.shape[0]
    out = prod[::sum, :, :, :] / n_run
    return out


def compute_gn(res, n=2, verbose=True):
    # -- verbose ?
    def _p(x):
        if verbose:
            print(x)

    # -- prepare output
    out = {"r": None, "c": None}

    # -- correlated
    _p("+ compute correlated")
    t0 = time.time()
    # compute unnorm
    Gn = _Gn(res["c"], n)
    # norm
    n_run = res["c"].shape[0]
    norm = 1
    for i in range(n):
        norm = norm * res["d"] / n_run

    gn = Gn / norm
    # store
    out["c"] = gn
    tf = time.time()
    _p(f"FINISHED IN {tf-t0:.2f} s \n")

    # -- random
    if res["r"] is not None:
        _p("+ compute random")
        t0 = time.time()
        # compute unnorm
        Gn = _Gn(res["r"], n)
        # norm
        gn = Gn / norm
        # store
        out["r"] = gn
        tf = time.time()
        _p(f"FINISHED IN {tf-t0:.2f} s \n")

    return out


def compute_gn_vs_binsize(res, n=2, step=1, use_tqdm=True, weighted=True):

    # -- settings
    n_max = np.max(res["c"].shape[1:])
    rebin_list = np.arange(1, n_max, step)

    # -- prepare output
    out = {
        "gc": np.zeros_like(rebin_list, dtype=float),
        "gc_err": np.zeros_like(rebin_list, dtype=float),
        "gr": np.zeros_like(rebin_list, dtype=float),
        "gr_err": np.zeros_like(rebin_list, dtype=float),
        "dk": np.zeros_like(rebin_list, dtype=float),
    }

    # -- compute
    if use_tqdm:
        iterator = enumerate(tqdm(rebin_list))
    else:
        iterator = enumerate(rebin_list)

    for i, nbin in iterator:
        # bin
        rbin = {
            "c": res["c"][:, :: bh.rebin(nbin), :: bh.rebin(nbin), :: bh.rebin(nbin)],
            "r": res["r"][:, :: bh.rebin(nbin), :: bh.rebin(nbin), :: bh.rebin(nbin)],
            "d": res["d"][:: bh.rebin(nbin), :: bh.rebin(nbin), :: bh.rebin(nbin)],
        }
        # compute
        gbin = compute_gn(rbin, n, verbose=False)
        # weight ?
        if weighted:
            weights = rbin["d"]
        else:
            weights = None
        # store
        k_grid = gbin["c"].axes[0].edges
        out["dk"][i] = k_grid[1] - k_grid[0]
        for k in ["c", "r"]:
            av = np.average(gbin[k], weights=weights)
            var = np.average((gbin[k] - av) ** 2, weights=weights)
            out[f"g{k}"][i] = av
            out[f"g{k}_err"][i] = np.sqrt(var) / np.sqrt(gbin[k].size)

    return out


# %% DISTRIBUTION FUNCTIONS
"""Functions to compute counts distributions P(n) from the density histograms"""

# %% PLOTTING UTILITY FUNCTIONS
"""Functions to plot the histograms"""


def plothist2d(ax, h, **params):
    return ax.pcolormesh(*h.axes.edges.T, h.view().T, **params)


def plot_g_vs_radius(g, ax, d=None, **params):
    # -- settings from histogram
    kmax = g.axes[0].edges.max()
    dk = np.mean(np.diff(g.axes[0].edges))
    n_points = g.shape[0]
    klist = np.linspace(dk, kmax, n_points)
    # -- prepare lists
    x = np.zeros_like(klist, dtype=float)
    err = np.zeros_like(klist, dtype=float)
    # -- compute
    for i, kr in enumerate(klist):
        start = bh.loc(-kr)
        stop = bh.loc(kr)
        gslice = g[start:stop, start:stop, start:stop]
        if d is not None:
            dslice = d[start:stop, start:stop, start:stop]
        else:
            dslice = None
        x[i] = np.average(gslice, weights=dslice)
        variance = np.average((gslice - x[i]) ** 2, weights=dslice)
        err[i] = np.sqrt(variance) / np.sqrt(gslice.size)

    # -- plot
    ax.errorbar(klist, x, yerr=err, **params)
    return klist, x, err


def plot_g2_cuts(g2, fig, ax, cut, w=0, **params):
    c = bh.loc(cut)
    len_cut = g2["c"][c - w : c + w + 1, :, :].shape[0]
    print(len_cut)
    for i_ax, g in enumerate([g2["c"], g2["r"]]):
        # x
        cax = ax[i_ax, 0]
        pcm = plothist2d(cax, g[c - w : c + w + 1 : bh.sum, :, :] / len_cut, **params)
        fig.colorbar(pcm, ax=cax)
        # y
        cax = ax[i_ax, 1]
        pcm = plothist2d(cax, g[:, c - w : c + w + 1 : bh.sum, :] / len_cut, **params)
        fig.colorbar(pcm, ax=cax)
        # z
        cax = ax[i_ax, 2]
        pcm = plothist2d(cax, g[:, :, c - w : c + w + 1 : bh.sum] / len_cut, **params)
        fig.colorbar(pcm, ax=cax)
    for i in range(3):
        ax[0, i].set_title("corr")
        ax[1, i].set_title("rand")
