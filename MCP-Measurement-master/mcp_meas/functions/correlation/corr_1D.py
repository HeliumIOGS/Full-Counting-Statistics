# -*- coding: utf-8 -*-
"""
Author   : alex
Created  : 2021-02-02 15:04:14
Modified : 2021-06-07 16:31:52

Comments : (1D) Correlation calculation functions
"""

# %% IMPORTS

# -- global

import numpy as np
from numba import jit
from tqdm import tqdm

# %% FUNCTIONS

# == util


def comp_G(data_list, pars, Gfunc, xyz=True, n_max=None):
    """
    Batch calculation of the Gfunc correlation function.
    data_list must be a list :
        data_list = [(kx_1, ky_1, kz_1), ... , (kx_N, ky_N, kz_N)]
    where kx_1 is a list (array-like) of momentum, corresponding to run #1

    Gfunc must return : hist, bins

    Where hist is either the 1D correlation function (if xyz set to False)
    or a dictionnary (hist = {"x":hist_x, "y":hist_y, "z":hist_z}) of the
    correlations along x, y and z (if xyz set to True)
    """
    hist = 0
    N = []
    for k in tqdm(data_list[:n_max]):
        kx, ky, kz = k
        n_hist, bins = Gfunc(kx, ky, kz, **pars)
        if xyz:
            hist = hist + n_hist["x"] + n_hist["y"] + n_hist["z"]
        else:
            hist = hist + n_hist
        N.append(len(kx))
    N = np.array(N)
    return hist, bins, N


# == local correlations


@jit
def G2_local(kx, ky, kz, dk_max, dk_bin, dk_trans):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist_x = np.zeros(N_bins, dtype=np.uint64)
    hist_y = np.zeros(N_bins, dtype=np.uint64)
    hist_z = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            # diffs
            dkx = np.abs(kx[i] - kx[j])
            dky = np.abs(ky[i] - ky[j])
            dkz = np.abs(kz[i] - kz[j])
            # x
            if dkx < dk_max and dky < dk_trans and dkz < dk_trans:
                hist_x[int(dkx / dk_bin)] += 1
            # y
            if dky < dk_max and dkx < dk_trans and dkz < dk_trans:
                hist_y[int(dky / dk_bin)] += 1
            # z
            if dkz < dk_max and dky < dk_trans and dkx < dk_trans:
                hist_z[int(dkz / dk_bin)] += 1

    hist = {"x": hist_x, "y": hist_y, "z": hist_z}
    return hist, bins


@jit
def G3_local(kx, ky, kz, dk_max, dk_bin, dk_trans):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist_x = np.zeros(N_bins, dtype=np.uint64)
    hist_y = np.zeros(N_bins, dtype=np.uint64)
    hist_z = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                # diffs
                # i-j
                dkx_1 = np.abs(kx[i] - kx[j])
                dky_1 = np.abs(ky[i] - ky[j])
                dkz_1 = np.abs(kz[i] - kz[j])
                # i-k
                dkx_2 = np.abs(kx[i] - kx[k])
                dky_2 = np.abs(ky[i] - ky[k])
                dkz_2 = np.abs(kz[i] - kz[k])
                # diffs
                dkx_12 = np.abs(dkx_1 - dkx_2)
                dky_12 = np.abs(dky_1 - dky_2)
                dkz_12 = np.abs(dkz_1 - dkz_2)

                # x
                if (
                    dkx_1 < dk_max
                    and dky_1 < dk_trans
                    and dkz_1 < dk_trans
                    and dky_2 < dk_trans
                    and dkz_2 < dk_trans
                    and dkx_12 < dk_bin
                ):
                    hist_x[int(dkx_1 / dk_bin)] += 1
                # y
                if (
                    dky_1 < dk_max
                    and dkx_1 < dk_trans
                    and dkz_1 < dk_trans
                    and dkx_2 < dk_trans
                    and dkz_2 < dk_trans
                    and dky_12 < dk_bin
                ):
                    hist_y[int(dky_1 / dk_bin)] += 1
                # z
                if (
                    dkz_1 < dk_max
                    and dky_1 < dk_trans
                    and dkx_1 < dk_trans
                    and dky_2 < dk_trans
                    and dkx_2 < dk_trans
                    and dkz_12 < dk_bin
                ):
                    hist_z[int(dkz_1 / dk_bin)] += 1

    hist = {"x": hist_x, "y": hist_y, "z": hist_z}
    return hist, bins


@jit
def G4_local(kx, ky, kz, dk_max, dk_bin, dk_trans):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist_x = np.zeros(N_bins, dtype=np.uint64)
    hist_y = np.zeros(N_bins, dtype=np.uint64)
    hist_z = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                for p in range(k):
                    # diffs
                    # i-j
                    dkx_1 = np.abs(kx[i] - kx[j])
                    dky_1 = np.abs(ky[i] - ky[j])
                    dkz_1 = np.abs(kz[i] - kz[j])
                    # i-k
                    dkx_2 = np.abs(kx[i] - kx[k])
                    dky_2 = np.abs(ky[i] - ky[k])
                    dkz_2 = np.abs(kz[i] - kz[k])
                    # i-p
                    dkx_3 = np.abs(kx[i] - kx[p])
                    dky_3 = np.abs(ky[i] - ky[p])
                    dkz_3 = np.abs(kz[i] - kz[p])
                    # diffs diffs
                    # 1-2
                    dkx_12 = np.abs(dkx_1 - dkx_2)
                    dky_12 = np.abs(dky_1 - dky_2)
                    dkz_12 = np.abs(dkz_1 - dkz_2)
                    # 1-3
                    dkx_13 = np.abs(dkx_1 - dkx_3)
                    dky_13 = np.abs(dky_1 - dky_3)
                    dkz_13 = np.abs(dkz_1 - dkz_3)

                    # x
                    if (
                        dkx_1 < dk_max
                        and dky_1 < dk_trans
                        and dkz_1 < dk_trans
                        and dky_2 < dk_trans
                        and dkz_2 < dk_trans
                        and dky_3 < dk_trans
                        and dkz_3 < dk_trans
                        and dkx_13 < dk_bin
                        and dkx_12 < dk_bin
                    ):
                        hist_x[int(dkx_1 / dk_bin)] += 1
                    # y
                    if (
                        dky_1 < dk_max
                        and dkx_1 < dk_trans
                        and dkz_1 < dk_trans
                        and dkx_2 < dk_trans
                        and dkz_2 < dk_trans
                        and dkx_3 < dk_trans
                        and dkz_3 < dk_trans
                        and dky_13 < dk_bin
                        and dky_12 < dk_bin
                    ):
                        hist_y[int(dky_1 / dk_bin)] += 1
                    # z
                    if (
                        dkz_1 < dk_max
                        and dkx_1 < dk_trans
                        and dky_1 < dk_trans
                        and dkx_2 < dk_trans
                        and dky_2 < dk_trans
                        and dky_3 < dk_trans
                        and dky_3 < dk_trans
                        and dkz_13 < dk_bin
                        and dkz_12 < dk_bin
                    ):
                        hist_z[int(dkz_1 / dk_bin)] += 1
    hist = {"x": hist_x, "y": hist_y, "z": hist_z}
    return hist, bins


@jit
def G4_local_test(kx, ky, kz, dk_max, dk_bin, dk_trans):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    N_counts = 0
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                for p in range(k):
                    # diffs
                    # i-j
                    dkx_1 = np.abs(kx[i] - kx[j])
                    dky_1 = np.abs(ky[i] - ky[j])
                    dkz_1 = np.abs(kz[i] - kz[j])
                    # i-k
                    dkx_2 = np.abs(kx[i] - kx[k])
                    dky_2 = np.abs(ky[i] - ky[k])
                    dkz_2 = np.abs(kz[i] - kz[k])
                    # i-p
                    dkx_3 = np.abs(kx[i] - kx[p])
                    dky_3 = np.abs(ky[i] - ky[p])
                    dkz_3 = np.abs(kz[i] - kz[p])
                    # diffs diffs
                    # 1-2
                    dkx_12 = np.abs(dkx_1 - dkx_2)
                    dky_12 = np.abs(dky_1 - dky_2)
                    dkz_12 = np.abs(dkz_1 - dkz_2)
                    dk_12 = np.sqrt(dkx_12 ** 2 + dky_12 ** 2 + dkz_12 ** 2)
                    # 1-3
                    dkx_13 = np.abs(dkx_1 - dkx_3)
                    dky_13 = np.abs(dky_1 - dky_3)
                    dkz_13 = np.abs(dkz_1 - dkz_3)
                    dk_13 = np.sqrt(dkx_13 ** 2 + dky_13 ** 2 + dkz_13 ** 2)

                    dk = np.sqrt(dk_12 ** 2 + dk_13 ** 2)
                    if dk < dk_max:
                        hist[int(dk / dk_bin)] += 1

    return hist, bins


# ---- OLD


@jit
def G3_local_old(kx, ky, kz, dk_max, dk_bin, dk_trans):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist_x = np.zeros(N_bins, dtype=np.uint64)
    hist_y = np.zeros(N_bins, dtype=np.uint64)
    hist_z = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                # diffs
                # i-j
                dkx_1 = np.abs(kx[i] - kx[j])
                dky_1 = np.abs(ky[i] - ky[j])
                dkz_1 = np.abs(kz[i] - kz[j])
                # i-k
                dkx_2 = np.abs(kx[i] - kx[k])
                dky_2 = np.abs(ky[i] - ky[k])
                dkz_2 = np.abs(kz[i] - kz[k])
                # sums
                dkx = dkx_1 + dkx_2
                dky = dky_1 + dky_2
                dkz = dkz_1 + dkz_2
                # x
                if (
                    dkx < dk_max
                    and dky_1 < dk_trans
                    and dkz_1 < dk_trans
                    and dky_2 < dk_trans
                    and dkz_2 < dk_trans
                ):
                    hist_x[int(dkx / dk_bin)] += 1
                # y
                if (
                    dky < dk_max
                    and dkx_1 < dk_trans
                    and dkz_1 < dk_trans
                    and dkx_2 < dk_trans
                    and dkz_2 < dk_trans
                ):
                    hist_y[int(dky / dk_bin)] += 1
                # z
                if (
                    dkz < dk_max
                    and dky_1 < dk_trans
                    and dkx_1 < dk_trans
                    and dky_2 < dk_trans
                    and dkx_2 < dk_trans
                ):
                    hist_x[int(dkz / dk_bin)] += 1
    hist = {"x": hist_x, "y": hist_y, "z": hist_z}
    return hist, bins


@jit
def G4_local_old(kx, ky, kz, dk_max, dk_bin, dk_trans):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist_x = np.zeros(N_bins, dtype=np.uint64)
    hist_y = np.zeros(N_bins, dtype=np.uint64)
    hist_z = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                for p in range(k):
                    # diffs
                    # i-j
                    dkx_1 = np.abs(kx[i] - kx[j])
                    dky_1 = np.abs(ky[i] - ky[j])
                    dkz_1 = np.abs(kz[i] - kz[j])
                    # i-k
                    dkx_2 = np.abs(kx[i] - kx[k])
                    dky_2 = np.abs(ky[i] - ky[k])
                    dkz_2 = np.abs(kz[i] - kz[k])
                    # i-p
                    dkx_3 = np.abs(kx[i] - kx[p])
                    dky_3 = np.abs(ky[i] - ky[p])
                    dkz_3 = np.abs(kz[i] - kz[p])
                    # sums
                    dkx = dkx_1 + dkx_2 + dkx_3
                    dky = dky_1 + dky_2 + dky_3
                    dkz = dkz_1 + dkz_2 + dkz_3
                    # x
                    if (
                        dkx < dk_max
                        and dky_1 < dk_trans
                        and dkz_1 < dk_trans
                        and dky_2 < dk_trans
                        and dkz_2 < dk_trans
                        and dky_3 < dk_trans
                        and dkz_3 < dk_trans
                    ):
                        hist_x[int(dkx / dk_bin)] += 1
                    # y
                    if (
                        dky < dk_max
                        and dkx_1 < dk_trans
                        and dkz_1 < dk_trans
                        and dkx_2 < dk_trans
                        and dkz_2 < dk_trans
                        and dkx_3 < dk_trans
                        and dkz_3 < dk_trans
                    ):
                        hist_y[int(dky / dk_bin)] += 1
                    # z
                    if (
                        dkz < dk_max
                        and dky_1 < dk_trans
                        and dkx_1 < dk_trans
                        and dky_2 < dk_trans
                        and dkx_2 < dk_trans
                        and dky_3 < dk_trans
                        and dkx_3 < dk_trans
                    ):
                        hist_x[int(dkz / dk_bin)] += 1
    hist = {"x": hist_x, "y": hist_y, "z": hist_z}

    return hist, bins


@jit
def G5_local(kx, ky, kz, dk_max, dk_bin, dk_trans):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist_x = np.zeros(N_bins, dtype=np.uint64)
    hist_y = np.zeros(N_bins, dtype=np.uint64)
    hist_z = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                for p in range(k):
                    for q in range(p):
                        # diffs
                        # i-j
                        dkx_1 = np.abs(kx[i] - kx[j])
                        dky_1 = np.abs(ky[i] - ky[j])
                        dkz_1 = np.abs(kz[i] - kz[j])
                        # i-k
                        dkx_2 = np.abs(kx[i] - kx[k])
                        dky_2 = np.abs(ky[i] - ky[k])
                        dkz_2 = np.abs(kz[i] - kz[k])
                        # i-p
                        dkx_3 = np.abs(kx[i] - kx[p])
                        dky_3 = np.abs(ky[i] - ky[p])
                        dkz_3 = np.abs(kz[i] - kz[p])
                        # i-q
                        dkx_4 = np.abs(kx[i] - kx[q])
                        dky_4 = np.abs(ky[i] - ky[q])
                        dkz_4 = np.abs(kz[i] - kz[q])
                        # sums
                        dkx = dkx_1 + dkx_2 + dkx_3 + dkx_4
                        dky = dky_1 + dky_2 + dky_3 + dky_4
                        dkz = dkz_1 + dkz_2 + dkz_3 + dkz_4
                        # x
                        if (
                            dkx < dk_max
                            and dky_1 < dk_trans
                            and dkz_1 < dk_trans
                            and dky_2 < dk_trans
                            and dkz_2 < dk_trans
                            and dky_3 < dk_trans
                            and dkz_3 < dk_trans
                            and dky_4 < dk_trans
                            and dkz_4 < dk_trans
                        ):
                            hist_x[int(dkx / dk_bin)] += 1
                        # y
                        if (
                            dky < dk_max
                            and dkx_1 < dk_trans
                            and dkz_1 < dk_trans
                            and dkx_2 < dk_trans
                            and dkz_2 < dk_trans
                            and dkx_3 < dk_trans
                            and dkz_3 < dk_trans
                            and dkx_4 < dk_trans
                            and dkz_4 < dk_trans
                        ):
                            hist_y[int(dky / dk_bin)] += 1
                        # z
                        if (
                            dkz < dk_max
                            and dky_1 < dk_trans
                            and dkx_1 < dk_trans
                            and dky_2 < dk_trans
                            and dkx_2 < dk_trans
                            and dky_3 < dk_trans
                            and dkx_3 < dk_trans
                            and dky_4 < dk_trans
                            and dkx_4 < dk_trans
                        ):
                            hist_x[int(dkz / dk_bin)] += 1
    hist = {"x": hist_x, "y": hist_y, "z": hist_z}

    return hist, bins


# == "k/-k" correlations


@jit
def G2_kmk(kx, ky, kz, dk_max, dk_bin):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            dkx = kx[i] + kx[j]
            dky = ky[i] + ky[j]
            dkz = kz[i] + kz[j]
            dk = np.sqrt(dkx ** 2 + dky ** 2 + dkz ** 2)
            if dk < dk_max:
                hist[int(dk / dk_bin)] += 1
    return hist, bins


@jit
def G3_kmk(kx, ky, kz, dk_max, dk_bin):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                dkx = kx[i] + kx[j] + kx[k]
                dky = ky[i] + ky[j] + ky[k]
                dkz = kz[i] + kz[j] + kz[k]
                dk = np.sqrt(dkx ** 2 + dky ** 2 + dkz ** 2)
                if dk < dk_max:
                    hist[int(dk / dk_bin)] += 1
    return hist, bins


@jit
def G3_test(kx, ky, kz, dk_max, dk_bin):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                dkx = kx[i] + kx[j] - kx[k]
                dky = ky[i] + ky[j] - ky[k]
                dkz = kz[i] + kz[j] - kz[k]
                dk = np.sqrt(dkx ** 2 + dky ** 2 + dkz ** 2)
                if dk < dk_max:
                    hist[int(dk / dk_bin)] += 1
    return hist, bins


@jit
def G4_kmk(kx, ky, kz, dk_max, dk_bin):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                for p in range(k):
                    dkx = kx[i] + kx[j] + kx[k] + kx[p]
                    dky = ky[i] + ky[j] + ky[k] + ky[p]
                    dkz = kz[i] + kz[j] + kz[k] + kz[p]
                    dk = np.sqrt(dkx ** 2 + dky ** 2 + dkz ** 2)
                    if dk < dk_max:
                        hist[int(dk / dk_bin)] += 1
    return hist, bins


@jit
def G5_kmk(kx, ky, kz, dk_max, dk_bin):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                for p in range(k):
                    for q in range(p):
                        dkx = kx[i] + kx[j] + kx[k] + kx[p] + kx[q]
                        dky = ky[i] + ky[j] + ky[k] + ky[p] + ky[q]
                        dkz = kz[i] + kz[j] + kz[k] + kz[p] + kz[q]
                        dk = np.sqrt(dkx ** 2 + dky ** 2 + dkz ** 2)
                        if dk < dk_max:
                            hist[int(dk / dk_bin)] += 1
    return hist, bins


@jit
def G6_kmk(kx, ky, kz, dk_max, dk_bin):
    # -- prepare bins
    N_bins = int(dk_max / dk_bin)
    hist = np.zeros(N_bins, dtype=np.uint64)
    bins = np.arange(N_bins) * dk_bin
    # -- correlate
    N_atoms = len(kx)
    for i in range(N_atoms):
        for j in range(i):
            for k in range(j):
                for p in range(k):
                    for q in range(p):
                        for r in range(q):
                            dkx = kx[i] + kx[j] + kx[k] + kx[p] + kx[q] + kx[r]
                            dky = ky[i] + ky[j] + ky[k] + ky[p] + ky[q] + ky[r]
                            dkz = kz[i] + kz[j] + kz[k] + kz[p] + kz[q] + kz[r]
                            dk = np.sqrt(dkx ** 2 + dky ** 2 + dkz ** 2)
                            if dk < dk_max:
                                hist[int(dk / dk_bin)] += 1
    return hist, bins
