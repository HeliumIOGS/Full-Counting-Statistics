#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:46:10 2021

@author: JP Bureik

Base class for all processing of MCP measurement data.

Select data in terms of atom number fluctuations, recenter and rotate. Save as
DataFrame of momenta in k-space.
"""

# Standard library imports:

from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd


class McpMeasProcess:

    def __init__(self, data_dirpath):

        self.data_dirpath = data_dirpath

    """
    PUBLIC METHODS
    """

    def load_reconstr_data(self):

        """ Load .TDC (text-)files of reconstructed data into DataFrame. """
        # Create array of filepaths for each data file
        data_filepath = []
        data_filename = [f for f in listdir(self.data_dirpath) if
                         isfile(join(self.data_dirpath, f))]
        for k in range(len(data_filename)):
            data_filepath.append(self.data_dirpath + '/' + data_filename[k])
        # Alphabetical sorting = chronological sorting for correct file names:
        data_filepath.sort()
        # Load data and append to DataFrame shot by shot:
        df_raw_cols = ['mcp_x', 'mcp_y', 'mcp_z']
        self.df_raw = pd.DataFrame(columns=df_raw_cols)
        for file in data_filepath:
            with open(file, 'r') as f:
                """ Single shot -> list w/ each entry = str(x,y,z-coordinates
                of 1 atom, D_Q, D(x,y)):"""
                single_shot_str = f.readlines()
                # List of x,y,z coordinate lists for a single shot:
                single_shot_coords = [[], [], []]
                # Convert str to float:
                for atom_coords_str in single_shot_str:
                    atom_coords_str = atom_coords_str.replace(
                        '\n', '').split(' ')
                    atom_coords = [float(coord_str) for coord_str in
                                   atom_coords_str]
                    # Discard D_Q and D(x,y), use only first 3 values:
                    for i in range(3):
                        single_shot_coords[i].append(atom_coords[i])
                # 1 run -> 1 row w/ [[x], [y], [z]] coords of all atoms in run:
                self.df_raw = self.df_raw.append({df_raw_cols[i]:
                                                 single_shot_coords[i] for i in
                                                 range(3)}, ignore_index=True)
        self.df_raw.index.rename('Run', inplace=True)
        # Get statistics on loaded data:
        self.tot_file_nb = len(self.df_raw)
        self.nb_of_atoms = [len(self.df_raw.iloc[i]['mcp_x']) for i in
                            range(len(self.df_raw))]
        self.mean_at_nb = np.mean(self.nb_of_atoms)

    def recenter_data(self):

        """ Recenter the loaded data. """
        pass

    def rotate_data(self):

        """ Rotate the recentered data to align from MCP to lattice axes. """
        pass

    def momentum_conv(self):

        """ Convert the rotated data to momentum values. """
        pass

    def save_data(self):

        """ Save converted data to hdf5. """
        pass

    """
    PRIVATE METHODS
    """
