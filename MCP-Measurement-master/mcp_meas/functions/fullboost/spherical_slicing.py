# -*- coding: utf-8 -*-
"""
Author   : Alexandre
Created  : 2021-06-16 11:09:31
Modified : 2021-06-16 11:25:13

Comments :
"""

# %% IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin


# %% COORDINATES TRANSFORMATION


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


# %% LOW-LEVEL FUNCTIONS


def theta_volumes(theta_grid):
    """compute the relative volumes of spherical elements with a given theta grid"""
    theta_grid = np.sort(theta_grid)
    N = len(theta_grid) - 1
    volumes = np.zeros(N)
    for i in range(N):
        v = np.cos(theta_grid[i]) - np.cos(theta_grid[i + 1])
        volumes[i] = v
    return volumes


def real_theta_volumes(theta_grid, phi_grid, r0, r1):
    """compute the volumes of sperical volume elements, between the radiuses
    `r0` and `r1`, with a grid of polar angles `theta_grid` and a grid of
    azimuthal angles `phi_grid`"""
    theta_grid = np.sort(theta_grid)
    theta_vol = -np.diff(np.cos(theta_grid))
    dphi = np.diff(phi_grid)[0]
    volumes = theta_vol * dphi * (r1 ** 3 - r0 ** 3) / 3
    return volumes


def min_func(theta_grid):
    """optimization function used in `optimum_theta_spacing`"""
    # -- prepare full grid
    # add 0 and pi, and make sure it starts with 0 and ends with pi
    theta_grid = np.concatenate(([0], theta_grid, [np.pi]))
    theta_grid = np.sort(theta_grid)
    # if not, return a HUGE cost !
    if theta_grid[0] < 0 or theta_grid[-1] > np.pi:
        return np.inf
    # -- compute volumes
    volumes = theta_volumes(theta_grid)
    # -- return : std of volumes
    return np.std(volumes)


def optimum_theta_spacing(N):
    """finds an optimimum theta grid to divide a sperical shell into `N` 'boxes'
    of ~equal volume"""
    # guess : evenly spaced theta
    guess = np.linspace(0, np.pi, N + 1)
    # remove 0 and pi (added by min_func)
    guess = guess[1:-1]
    # optimize
    theta_grid = fmin(min_func, x0=guess, disp=False)
    theta_grid = np.sort(theta_grid)
    # add 0 and pi
    theta_grid = np.concatenate(([0], theta_grid, [np.pi]))
    return theta_grid


# %% MAIN FUNCTION : SHELL SLICING


# == COMPUTE


def shell_slicing_3D(r0, r1):
    """computes a grid of polar angles `theta_grid` and azimuthal angles `phi_grid`
    to divide a spherical shell between `r0` and `r1` into a collection of ~isovolumes
    boxes. The target volume is the one of the core shell, i.e. from r=0 to r=dr=r1-r0
    """
    # compute target volume
    dr = r1 - r0
    v0 = (4 / 3) * np.pi * dr ** 3
    # target boxes number
    vshell = (4 / 3) * np.pi * (r1 ** 3 - r0 ** 3)
    Nshell = np.round(vshell / v0)
    # target Nphi and Ntheta
    Nphi = int(np.round(np.sqrt(2 * Nshell)))
    Ntheta = int(np.round(Nphi / 2))
    # optimize theta spacing
    theta_grid = optimum_theta_spacing(Ntheta)
    phi_grid = np.linspace(0, 2 * np.pi, Nphi + 1)
    volumes = theta_volumes(theta_grid)
    # result
    result = {
        "theta_grid": theta_grid,
        "phi_grid": phi_grid,
        "Nphi": Nphi,
        "Ntheta": Ntheta,
        "volumes": volumes,
        "r0": r0,
        "r1": r1,
        "ri": 0.5 * (r0 + r1),
    }
    return result


# == PLOT


def plot_shell_slicing(result, figsize=(10, 10)):
    # get results
    theta_grid = result["theta_grid"]
    phi_grid = result["phi_grid"]

    # prepare grid
    phi = np.linspace(0, 2 * np.pi, 200)
    theta = np.linspace(0, np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)
    R = result["ri"]
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    COLORS = np.empty_like(X, dtype=str)
    clist = ["b", "w", "r", "k", "c", "y", "m"]
    clist = ["b", "w", "r", "w"]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = THETA[i, j]
            phi = PHI[i, j]
            i_theta = np.searchsorted(theta_grid, theta)
            if i_theta == 0:
                i_theta = 1
            i_phi = np.searchsorted(phi_grid, phi)
            if i_phi == 0:
                i_phi = 1
            i_tot = i_phi + i_theta
            COLORS[i, j] = clist[i_tot % len(clist)]
            pass
    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    i_tot = 0
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=COLORS,
    )

    plt.show()
