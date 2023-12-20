# -*- coding: utf-8 -*-
"""
Author   : alex
Created  : 2021-02-05 10:11:15
Modified : 2021-06-07 10:09:30

Comments : Creates datasets (cocktails) from (previously processed)
           MCP data. This module works together with the mcp_process module
           (or the ~ equivalent data_maker MATLAB script). The job
           repartition is the following :

                + mcp_process : - loads .tdc data
                                - rotates / re-center data
                                - post-select on atom number
                    => saves momenta distributions run / run

                + mp_data_cocktail : - loads "mcp_process" output
                                     - filter impulsion zones
                                     - folds brillouin zones
                                     - can shuffle data / recombine data
                    => outputs / saves sets of momenta distribution
                       (each set can / must be analyzed independently)

            Note : the cocktail_maker steps are all optionnal, and can be
                   performed in an arbitrary order
"""

# %% IMPORTS

# -- global

import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import io


# %% FUNCTIONS


def filter_momentum(kx, ky, kz, type="box", k_shift=(0, 0, 0), **options):
    # -- check types
    implemented_types = ["box", "sphere"]
    msg = "implemented filter types : '" + "', '".join(implemented_types)
    assert type in implemented_types, msg

    # -- shift
    dkx, dky, dkz = k_shift
    kx_shift = kx - dkx
    ky_shift = ky - dky
    kz_shift = kz - dkz

    # -- box type
    if type == "box":
        # check options
        req_opt = ["k_min", "k_max"]
        msg = "'box' filter requires options : '" + "', '".join(req_opt)
        assert all(opt in options for opt in req_opt), msg
        # get options
        k_min = options["k_min"]
        k_max = options["k_max"]
        # filter
        mask_in = (
            (np.abs(kx_shift) < k_min)
            * (np.abs(ky_shift) < k_min)
            * (np.abs(kz_shift) < k_min)
        )

        mask_out = (
            (np.abs(kx_shift) < k_max)
            * (np.abs(ky_shift) < k_max)
            * (np.abs(kz_shift) < k_max)
        )
        mask = mask_out * ~mask_in

    # -- sphere type
    if type == "sphere":
        # check options
        req_opt = ["k_min", "k_max"]
        msg = "'sphere' filter requires options : '" + "', '".join(req_opt)
        assert all(opt in options for opt in req_opt), msg
        # get options
        k_min = options["k_min"]
        k_max = options["k_max"]
        # filter
        k2 = kx_shift ** 2 + ky_shift ** 2 + kz_shift ** 2
        mask = (k2 > k_min ** 2) * (k2 < k_max ** 2)

    return mask


def fold_brillouin(kx, ky, kz):
    # right
    kx[kx > 0.5] -= 1
    ky[ky > 0.5] -= 1
    kz[kz > 0.5] -= 1
    # left
    kx[kx < -0.5] += 1
    ky[ky < -0.5] += 1
    kz[kz < -0.5] += 1

    return kx, ky, kz


# %% IMPLEMENT DATACOCKTAIL CLASS


class DataCocktail:
    """
    Generates data sets for further analysis
    """

    def __init__(self, data_root=".", data_set=None):
        """
        Object initialization, sets parameters
        """
        # public
        self.data_root = data_root
        self.data_set = data_set
        self.data = []
        self.verbose = False

        # private
        self._date_fmt = "%Y-%m-%d_%H%M%S"
        self._time_start = 0
        self._time_int = 0

    """
    PROPERTIES
    """

    @property
    def data_root(self):
        return self._data_root

    @data_root.setter
    def data_root(self, new_root):
        self._data_root = Path(new_root)

    """
    PUBLIC METHODS
    """

    # == DATA LOADING

    def load_data(self, format=None, lattice_axis=True):
        """
        Loads MCP data
        """
        # -- build data path
        data_path = self.data_root / self.data_set

        # -- check format
        format_list = {
            ".pkl": "panda_pkl",
            ".pkl4": "panda_pkl",
            ".mat": "matlab",
        }
        available_formats = {*format_list.values()}
        available_extensions = format_list.keys()
        if format is None:
            _, ext = os.path.splitext(data_path)
            if ext not in available_extensions:
                extension_str = "', '".join(available_extensions)
                print("Wrong file extension.")
                print("Available extensions are : '%s'" % extension_str)
                return
            else:
                format = format_list[ext]
        else:
            if format not in available_formats:
                format_str = "', '".join(available_formats)
                print("Wrong file format.")
                print("Available formats are : '%s'" % format_str)
                return

        # -- check that file exists
        if not data_path.is_file():
            print("Error loading %s : file does not exist" % data_path)
            return

        # -- load the data
        # - simple case : panda (pickle)
        if format == "panda_pkl":
            self.data = pd.read_pickle(data_path)
            self.data.rename(
                columns={"k_x": "kx", "k_y": "ky", "k_z": "kz"}, inplace=True
            )

        # - matlab file (from data_maker script)
        elif format == "matlab":
            if lattice_axis:
                col_offset=0
            else:
                col_offset=3
            # load .mat
            data = io.loadmat(data_path)
            # take momentum distribution
            k = data["momentum_lattice_axis"]
            # load into DataFrame:
            df_cols = ["kx", "ky", "kz"]
            df = pd.DataFrame()
            for i in range(len(df_cols)):
                df[df_cols[i]] = k[:, i+col_offset]
            df.index.rename("Run", inplace=True)
            # store
            self.data = df

        # -- verbose
        if self.verbose:
            self.display_data_info()

    def display_data_info(self):
        # -- get info
        if len(self.data) == 0:
            print("data is empty")
            return

        # atom number
        n_at_list = []
        for i in self.data.index:
            n_at_list.append(len(self.data.kx[i]))

        n_at_list = np.array(n_at_list)
        n_at_list = n_at_list[n_at_list > 2]
        # compute stats
        n_runs = len(self.data.index)
        n_empty_runs = n_runs - len(n_at_list)
        n_atoms = np.sum(n_at_list)
        data_path = self.data_root / self.data_set
        # -- get info
        print("+ file loaded : %s" % data_path)
        print("+ run number  : %i" % n_runs)
        print("+ atom number : %.2e" % n_atoms)
        print("+ empty runs  : %i" % n_empty_runs)
        print("+ atom / run  :")
        print("    > mean : %.2e" % np.mean(n_at_list))
        print("    > max  : %.2e" % np.max(n_at_list))
        print("    > min  : %.2e" % np.min(n_at_list))
        print("    > std  : %.2e" % np.std(n_at_list))

    def get_atom_number_stats(self):
        # atom number
        n_at_list = []
        for i in self.data.index:
            n_at_list.append(len(self.data.kx[i]))

        n_at_list = np.array(n_at_list)
        n_at_list = n_at_list[n_at_list > 2]

        out = {
            "mean": np.mean(n_at_list),
            "min": np.min(n_at_list),
            "max": np.max(n_at_list),
            "std": np.std(n_at_list),
        }
        return out

    # === EXPORT METHODS

    def copy_data(self):
        return self.data.copy(deep=True)

    def to_numpy(self):
        return self.data.to_numpy(copy=True)

    def to_dict(self):
        # prepare output dict
        dict = {}
        # get k as an array
        k_array = self.data.to_numpy(copy=True)
        # concatenate into one single array
        for i, k in enumerate(["kx", "ky", "kz"]):
            dict[k] = np.squeeze(np.concatenate(k_array[:, i]))
        # generate the "runs" array
        dict["run"] = np.concatenate(
            [np.ones(len(kx)) * i for i, kx in enumerate(k_array[:, 0])]
        )
        # return
        return dict

    def to_list(self):
        run_list = []
        for i in self._tqdm(self.data.index):
            kx = 1.0 * self.data.kx[i]
            ky = 1.0 * self.data.ky[i]
            kz = 1.0 * self.data.kz[i]
            run_list.append((kx, ky, kz))
        return run_list

    def save_data(
        self, filename="data_cocktail.pkl", root=None, format="pickle", **kwargs
    ):
        if root is None:
            root = self.data_root
        else:
            root = Path(root)
        save_path = root / filename
        if format == "pickle":
            self.data.to_pickle(save_path, **kwargs)
        elif format == "hdf":
            self.data.to_hdf(save_path, key="data", mode="w", **kwargs)
        else:
            print("> '%s' format not implemented" % format)
            return

        self._p("> saved as %s" % save_path)

    # == DATA PROCESSING

    def remove_empty_runs(self):
        self._tic()
        empty_runs = []
        for run in self.data.index:
            if len(self.data.kx[run]) < 2:
                empty_runs.append(run)
        self.data = self.data.drop(empty_runs)
        self._tac()

    def filter_momentum(self, type="box", **options):
        self._p("> filter momentum")
        self._tic()
        # -- if verbose : count initial atom number
        if self.verbose:
            n_atoms_before = 0
            for i in self.data.index:
                n_atoms_before += len(self.data.kx[i])

        # -- loop on all runs
        for run in self._tqdm(self.data.index):
            # get
            kx = self.data.kx[run]
            ky = self.data.ky[run]
            kz = self.data.kz[run]
            # mask
            mask = filter_momentum(kx, ky, kz, type, **options)
            # filter
            self.data.kx[run] = kx[mask]
            self.data.ky[run] = ky[mask]
            self.data.kz[run] = kz[mask]

        # -- if verbose : count final atom number
        if self.verbose:
            n_atoms_after = 0
            for i in self.data.index:
                n_atoms_after += len(self.data.kx[i])
            n_atoms_removed = n_atoms_before - n_atoms_after
            print("+ atoms ")
            print("    > before  : %.2e " % n_atoms_before)
            print("    > after   : %.2e " % n_atoms_after)
            print("    > removed : %.2e " % n_atoms_removed)

        self._tac()

    def fold_brillouin(self):
        self._p("> fold to first brillouin zone")
        self._tic()

        # -- loop on all runs
        for run in self._tqdm(self.data.index):
            # get
            kx = 1.0 * self.data.kx[run]
            ky = 1.0 * self.data.ky[run]
            kz = 1.0 * self.data.kz[run]
            # fold
            kx, ky, kz = fold_brillouin(kx, ky, kz)
            # save
            self.data.kx[run] = kx
            self.data.ky[run] = ky
            self.data.kz[run] = kz

        self._tac()

    def shuffle(self):
        self._p("> shuffle atoms (run per run)")
        self._tic()

        # -- loop on all runs
        for run in self._tqdm(self.data.index):
            # get
            kx = 1.0 * self.data.kx[run]
            ky = 1.0 * self.data.ky[run]
            kz = 1.0 * self.data.kz[run]
            # shuffle
            k = np.empty((len(kx), 3))
            k[:, 0] = np.squeeze(kx)
            k[:, 1] = np.squeeze(ky)
            k[:, 2] = np.squeeze(kz)
            np.random.shuffle(k)
            # store
            self.data.kx[run] = k[:, 0]
            self.data.ky[run] = k[:, 1]
            self.data.kz[run] = k[:, 2]
        self._tac()

    def recombine(self, n_runs):
        self._p("> recombine atoms")
        self._p("    > runs in  : %i" % len(self.data))
        self._p("    > runs out : %i" % n_runs)
        self._tic()

        # -- gather all
        self._p(" + gather all")
        # prepare empty list
        kx = []
        ky = []
        kz = []
        # loop on all runs
        for run in self._tqdm(self.data.index):
            kx.append(self.data.kx[run].squeeze())
            ky.append(self.data.ky[run].squeeze())
            kz.append(self.data.kz[run].squeeze())

        # convert to numpy array
        kx = np.concatenate(kx)
        ky = np.concatenate(ky)
        kz = np.concatenate(kz)

        # put in in one array
        k = np.empty((len(kx), 3))
        k[:, 0] = kx
        k[:, 1] = ky
        k[:, 2] = kz
        del kx, ky, kz

        # -- split
        self._p(" + split")
        k = np.array_split(k, n_runs)

        # -- load into DataFrame:
        self._p(" + load into dataframe")
        data_cols = ["kx", "ky", "kz"]
        data_recombined = pd.DataFrame(index=range(len(k)), columns=data_cols)
        time.sleep(0.2)
        for j in tqdm(range(len(k))):
            for i in range(len(data_cols)):
                data_recombined[data_cols[i]][j] = k[j][:, i]
        data_recombined.index.rename("Run", inplace=True)

        # -- save
        self.data = data_recombined
        self._tac()

    """
    PRIVATE METHODS
    """

    def _tic(self):
        # update time references
        self._time_start = time.time()
        self._time_int = time.time()

    def _toc(self):
        # get current time and compute durations
        current_time = time.time()
        step_duration = current_time - self._time_int
        total_duration = current_time - self._time_start
        # display
        msg = "DONE in %.2f s (total %.2f s)"
        self._p(msg % (step_duration, total_duration))
        # update
        self._time_int = current_time

    def _tac(self):
        # get current time and compute durations
        current_time = time.time()
        total_duration = current_time - self._time_start
        # display
        time.sleep(0.2)
        msg = "FINISHED in %.2f s "
        self._p(msg % total_duration)

    def _p(self, msg):
        if self.verbose:
            print(msg)
            time.sleep(0.2)

    def _tqdm(self, iterable):
        if self.verbose:
            return tqdm(iterable)
        else:
            return iterable
