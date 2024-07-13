import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# My libraries
from cpt import *
from mappings import color_mapping_layers
from lithology import LithologyPlotter
from voxel import Voxelizer

# %% Global Settings

savefig = True
output_path = 'figs'

# Revise for project settings class? Part of an overall parent class?

# %% Load CPT data

path = 'data\collar.csv'
collar_df = load_collar(path, savefig=savefig, output_folder=output_path)

# Load the processed CPT data
fname = r'./data/cpt-processed.xlsm'
sheetnames = ["Liq_22-01C",
              "Liq_22-02C",
              "Liq_22-03C",
              "Liq_22-04C",
              "Liq_22-05C",
              "Liq_22-06C",
              "Liq_22-07C",
              "Liq_22-08C",
              "Liq_22-09C",
              "Liq_22-10C",
              "Liq_22-11C",
              "Liq_22-12C"]

cpt_data, merged_df = load_cpt_data(fname, sheetnames)

# Revise for basic data class that gets ingested by CPT Plotter and Lithology Classes?
# Part of overall parent class?


# %% Plot CPT data

plt_soundings  = [['22-09C','22-10C','22-11C','22-12C'],
                  ['22-01C','22-02C','22-03C','22-04C','22-05C','22-06C'],
                  ['22-08C','22-07C'],
                  ['22-10C','22-03C','22-08C'],
                  ['22-09C','22-03C','22-08C'],
                  ['22-12C','22-05C','22-07C']
                  ]

plot_cpt_soundings(cpt_data, plt_soundings, plot_key='cpt-qc', plot_label="qc (tsf)",
                   color_mapping_layers=color_mapping_layers, xlim=(0, 250), 
                   savefig=savefig, output_folder=output_path)

plot_cpt_soundings(cpt_data, plt_soundings, plot_key='F', plot_label="Friction Ratio (%)",
                   color_mapping_layers=color_mapping_layers, xlim=(0, 15), 
                   savefig=savefig, output_folder=output_path)

# Revise to CPT Plotter class?


# %% Plot CPT lithology

collar_file = 'data/collar.csv'
projection_points = [(0,0),(1,1)]
plotter = LithologyPlotter(cpt_data, collar_file=collar_file,
                           pen_fac=0.2, fric_fac=0.3,
                           elev_3d=30, azim_3d=-60, radius_3d=0.05, 
                           point1=projection_points[0],
                           point2=projection_points[1],
                           savefig=savefig)

for plot_type in ['SBT','friction_ratio','penetration_resistance']:

    plotter.plot_type = plot_type
    plotter.update()
    plotter.plot(exts=['.svg', '.png'])
    

plotter.set_projection_line((0,1),(1,1))
plotter.plot(exts=['.svg', '.png'])
    
    
# Revise loading/preparng data? Accept data class?


# %% Voxelization

voxelizer = Voxelizer(plotter)
voxelizer.interpolate_voxel_lithology()
voxelizer.plot_voxel_lithology(exts=['.svg', '.png'])
voxelizer.report_voxel_volume()


plotter.plot_type = 'friction_ratio'
plotter.update()
voxelizer2 = Voxelizer(plotter)
voxelizer2.interpolate_voxel_lithology()
voxelizer2.plot_voxel_lithology(exts=['.svg', '.png'])
voxelizer2.report_voxel_volume()


plotter.plot_type = 'SBT'
plotter.update()
voxelizer3 = Voxelizer(plotter)
voxelizer3.interpolate_voxel_lithology()
voxelizer3.plot_voxel_lithology(exts=['.svg', '.png'])
voxelizer3.report_voxel_volume()



# Plot slices
voxelizer3.plot_voxel_slices_xy(plt_sounding=True, savefig=True, exts=['.svg', '.png'])
voxelizer3.plot_voxel_slices_xz(plt_sounding=True, savefig=True, exts=['.svg', '.png'], figsize=(40,20))
voxelizer3.plot_voxel_slices_yz(plt_sounding=True, savefig=True, exts=['.svg', '.png'], figsize=(40,20))

# for i in range(voxelizer.nz):
#     voxelizer3.plot_voxel_slice_xy(voxelizer.nz-i-1, plt_sounding=True)

# for i in range(voxelizer.ny):
#     voxelizer3.plot_voxel_slice_xz(i, plt_sounding=True)

# for i in range(voxelizer.nx):
#     voxelizer3.plot_voxel_slice_yz(i, plt_sounding=True)


# Revise to include downsampling of the sounding lithology interval?
# Revise to accept save paths?
# Revise to update plotter lithology from voxelizer?
# Right now I am copying the dataframe, but storing a pointer to original plotter object
# I think it is better to copy the data to voxelizer and not use plotter


# %%

# Add parent sounding class ?
# Add SPT borehole lithology ?
# Consider functional refactor, or avoiding use of self except at start/end of methods
#   to avoid side-effects
