import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.interpolate import griddata

from mappings import (
    color_mapping
)
from lithology import (
    LithologyPlotter
)


class Voxelizer:
        
    def __init__(self, plotter, Nvoxel=50):
        
        if not isinstance(plotter, LithologyPlotter):
            raise ValueError(f"Wrong type {type(plotter)}.")
    
        self.nx = self.ny = self.nz = Nvoxel # Init number of divisions in each dim
        self.plotter = plotter
        
        self.vox_df = plotter.lithology_df.copy()
        self.vox_collar_df = plotter.collar_df.copy()
        self.plot_type = plotter.plot_type
        
        self.grid_x = self.grid_y = self.grid_z = self.grid_values = None
        self.colors = None
        
    @property
    def get_plotter(self):
        return self.plotter
        
    @property
    def get_grid(self):
        return self.grid_x, self.grid_y, self.grid_z, self.grid_values
        
    def set_voxel_dimensions(self, Nvoxel=None, nx=None, ny=None, nz=None):
        """
        Set the number of divisions in each dimension
        """
        
        if nx is not None:
            self.nx = nx
        if ny is not None:
            self.ny = ny
        if nz is not None:
            self.nz = nz
        if Nvoxel is not None:
            self.nx = self.ny = self.nz = Nvoxel
        return
    
    def interpolate_voxel_lithology(self, Nvoxel=None, interp_method='nearest'):
        """
        Interpolate the voxel lithology from sounding lithology
        """
        
        # Create lithology code reference for plotting
        if self.plot_type == 'SBT':
            self.vox_df['lithology_code'] = self.vox_df[self.plot_type]-1 # Adjust for zero-based index
        else:
            self.vox_df['lithology_code'] = self.vox_df[self.plot_type]
            
        
        # Define the grid dimensions
        x = np.linspace(self.vox_collar_df['X'].min(), 
                        self.vox_collar_df['X'].max(), self.nx+1) # n+1 points for the vox corners
        y = np.linspace(self.vox_collar_df['Y'].min(), 
                        self.vox_collar_df['Y'].max(), self.ny+1) 
        z = np.linspace(self.vox_df['end_elevation'].min(), 
                        self.vox_df['start_elevation'].max(), self.nz+1)
        self.grid_x, self.grid_y, self.grid_z = np.meshgrid(x, y, z, indexing='ij')
        
        
        # Prepare the data for interpolation
        mid_elevations = (self.vox_df['start_elevation'] + self.vox_df['end_elevation']) / 2
        points = np.vstack((self.vox_collar_df.set_index('borehole_id').loc[self.vox_df['borehole_id'], 'X'].values,
                            self.vox_collar_df.set_index('borehole_id').loc[self.vox_df['borehole_id'], 'Y'].values,
                            mid_elevations)).T
        values = self.vox_df['lithology_code'].values  # Use numerical codes for interpolation
        
        # Interpolate lithology using griddata midpoints with nearest neighbor
        interp_x = (self.grid_x[:-1, :-1, :-1] + self.grid_x[1:, 1:, 1:]) / 2
        interp_y = (self.grid_y[:-1, :-1, :-1] + self.grid_y[1:, 1:, 1:]) / 2
        interp_z = (self.grid_z[:-1, :-1, :-1] + self.grid_z[1:, 1:, 1:]) / 2
        self.grid_values = griddata(points, values, (interp_x, interp_y, interp_z), method=interp_method)
        
        # Convert interpolated values to integers to match 
        self.grid_values = np.round(self.grid_values).astype(int)
        
        # Handle NaN values by filling with the nearest values
        self.grid_values = np.nan_to_num(self.grid_values, nan=np.nanmean(values)).astype(int)
        
        return
    
    def set_colors(self):
        """
        Set the voxel colors based on the lithology context
        """
        
        # Create a consistent color mapping for visualization
        unique_codes = np.unique(self.vox_df['lithology_code'])
        if self.plot_type == 'SBT':
            color_map = self.plotter.SBT_colors
        elif self.plot_type == 'friction_ratio':
            color_map = self.plotter.friction_ratio_colors
        elif self.plot_type == 'penetration_resistance':
            color_map = self.plotter.penetration_resistance_colors
        
        self.colors = np.zeros(self.grid_values.shape + (4,))
        for code in unique_codes:
            if self.plot_type == 'SBT':
                self.colors[self.grid_values == code] = color_map(code)
            else:
                self.colors[self.grid_values == code] = color_map(code / len(unique_codes))
            
        return unique_codes
        

    def plot_voxel_lithology(self, edgecolors=None, dpi=300, exts='.svg'):
        """
        Plot the voxel lithology results
        """
        if not isinstance(exts, list):
            exts = [exts]
        
        # Update the colors
        unique_codes = self.set_colors()
        
        # Update plotter settings
        self.plotter.show_plane=False
        self.plotter.radius_3d = 0.02
        
        # Plot the borehole lithology with the voxelization
        for code in unique_codes:
            
            # Create a boolean array for voxel plotting
            voxelarray = self.grid_values == code
            
            self.plotter.elev_3d=30
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111, projection='3d')
            ax.voxels(self.grid_x, self.grid_y, self.grid_z, voxelarray, 
                      facecolors=self.colors, edgecolors=edgecolors)
            self.plotter.create_3d_plot(ax)
            self.plotter.add_colorbar(fig, ax)
            for ext in exts:
                fname = f"voxel_borehole_{self.plot_type}_elev3d_{self.plotter.elev_3d}_{code}"+ext
                fig.savefig(fname, dpi=dpi, bbox_inches="tight")
            plt.show()
        
            self.plotter.elev_3d=0
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111, projection='3d')
            ax.voxels(self.grid_x, self.grid_y, self.grid_z, voxelarray, 
                      facecolors=self.colors, edgecolors=edgecolors)
            self.plotter.create_3d_plot(ax)
            self.plotter.add_colorbar(fig, ax)
            for ext in exts:
                fname = f"voxel_borehole_{self.plot_type}_elev3d_{self.plotter.elev_3d}_{code}"+ext
                fig.savefig(fname, dpi=dpi, bbox_inches="tight")
            plt.show()
        return
    
    def plot_voxel_slices_xy(self, figsize=(30,20), plt_sounding=False, savefig=False, dpi=300, exts=['.svg']):
        """
        Plot all XY plane voxel slices
        """
        # Update the colors
        _ = self.set_colors()
        
        ncols = 10
        nrows = int(np.ceil(self.nz/ncols))
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
        ax = ax.flatten()
        
        for i in range(self.nz):
            i = self.nz - 1 - i
            self.plot_voxel_slice_xy(i, ax=ax[i], plt_sounding=plt_sounding)
        
        if savefig:
            for ext in exts:
                fname = f"voxel_slice_{self.plot_type}__XY_slices"+ext
                fig.savefig(fname, dpi=dpi, bbox_inches='tight')
            
        plt.show()
        return
    
    
    def plot_voxel_slices_xz(self, figsize=(30,20), plt_sounding=False, savefig=False, dpi=300, exts=['.svg']):
        """
        Plot all XZ plane voxel slices
        """
        # Update the colors
        _ = self.set_colors()
        
        ncols = 10
        nrows = int(np.ceil(self.ny/ncols))
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
        ax = ax.flatten()
        
        for i in range(self.ny):
            i = self.ny - 1 - i
            
            self.plot_voxel_slice_xz(i, ax=ax[i], plt_sounding=plt_sounding)
            
        if savefig:
            for ext in exts:
                fname = f"voxel_slice_{self.plot_type}__XZ_slices"+ext
                fig.savefig(fname, dpi=dpi, bbox_inches='tight')
            
        plt.show()
        return
    
    def plot_voxel_slices_yz(self, figsize=(30,20), plt_sounding=False, savefig=False, dpi=300, exts=['.svg']):
        """
        Plot all YZ plane voxel slices
        """
        # Update the colors
        _ = self.set_colors()
        
        ncols = 10
        nrows = int(np.ceil(self.nx/ncols))
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
        ax = ax.flatten()
        
        for i in range(self.nx):
            i = self.nx - 1 - i
            
            self.plot_voxel_slice_yz(i, ax=ax[i], plt_sounding=plt_sounding)
        
        if savefig:
            for ext in exts:
                fname = f"voxel_slice_{self.plot_type}__YZ_slices"+ext
                fig.savefig(fname, dpi=dpi, bbox_inches='tight')
            
        plt.show()
        return
    
    
    def plot_voxel_slice_xy(self, i, ax=None, plt_sounding=False):
        """
        Plot XY plane voxel slice at index i
        """
        
        # Update the colors
        _ = self.set_colors()
        
        # Extract slice and grid extents
        slice_data = self.grid_values[:, :, i]
        slice_colors = self.colors[:, :, i]
        extents = [self.grid_x.min(), self.grid_x.max(), 
                   self.grid_y.min(), self.grid_y.max()]
        
        # Transpose the spatial dimensions (X, Y) to (Y, X) while preserving 
        # the RGBA channels (50, 50, 4) -> (50, 50, 4)
        # Note: if we used slice_data which is (50, 50), we could use .T
        slice_colors_transposed = np.transpose(slice_colors, (1, 0, 2))
        
        # Create new plot if no axes provided
        if ax is None:
            fig, ax = plt.subplots() 
            
        # Plot voxel slice
        ax.imshow(slice_colors_transposed, origin='lower', extent=extents, aspect='auto')
        
        # Overlay the borehole locations
        if plt_sounding:
            collar_df = self.plotter.get_collar.copy()
            ax.plot(collar_df['X'], collar_df['Y'], 'tab:red', ls='None', 
                     marker=r"$\bigoplus$", ms=10, mew=0.2)
            for j, row in collar_df.iterrows():
                ax.annotate(row['borehole_id'], (row['X'], row['Y']), fontsize=12, ha='right')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Slice {i}, Elev {self.grid_z[0,0,i]:.1f}')
        return
    
    def plot_voxel_slice_xz(self, i, ax=None, plt_sounding=False):
        """
        Plot XZ plane voxel slice at index i
        """
        
        # Update the colors
        _ = self.set_colors()
        
        # Extract slice and grid extents
        slice_data = self.grid_values[:, i, :]
        slice_colors = self.colors[:, i, :]
        extents = [self.grid_x.min(), self.grid_x.max(), 
                   self.grid_z.min(), self.grid_z.max()]
        
        # Transpose the spatial dimensions (X, Z) to (Z, X) while preserving 
        # the RGBA channels (50, 50, 4) -> (50, 50, 4)
        # Note: if we used slice_data which is (50, 50), we could use .T
        slice_colors_transposed = np.transpose(slice_colors, (1, 0, 2))
        
        # Create new plot if no axes provided
        if ax is None:
            fig, ax = plt.subplots() 
            
        # Plot voxel slice
        ax.imshow(slice_colors_transposed, origin='lower', extent=extents, aspect='auto')
        
        # Overlay the borehole locations
        if plt_sounding:
            collar_df = self.plotter.get_collar.copy()
            lithology_df = self.plotter.get_lithology.copy()
            group_df = lithology_df.groupby('borehole_id')
            
            for j, row in collar_df.iterrows():
                top = row['Z']
                bott = group_df.get_group(row['borehole_id']).end_elevation.min()
                ax.plot([row['X'], row['X']],[top, bott], '-', c='tab:red')
                ax.annotate(row['borehole_id'], (row['X'], row['Z']), fontsize=12, ha='right')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f'Slice {i}')
        return
    
    def plot_voxel_slice_yz(self, i, ax=None, plt_sounding=False):
        """
        Plot YZ plane voxel slice at index i
        """
        
        # Update the colors
        _ = self.set_colors()
        
        # Extract slice and grid extents
        slice_data = self.grid_values[i, :, :]
        slice_colors = self.colors[i, :, :]
        extents = [self.grid_y.min(), self.grid_y.max(), 
                   self.grid_z.min(), self.grid_z.max()]
        
        # Transpose the spatial dimensions (Y, Z) to (Z, Y) while preserving 
        # the RGBA channels (50, 50, 4) -> (50, 50, 4)
        # Note: if we used slice_data which is (50, 50), we could use .T
        slice_colors_transposed = np.transpose(slice_colors, (1, 0, 2))
        
        if ax is None:
            fig, ax = plt.subplots() 
            
        # Plot voxel slice
        ax.imshow(slice_colors_transposed, origin='lower', extent=extents, aspect='auto')
        
        # Overlay the borehole locations
        if plt_sounding:
            collar_df = self.plotter.get_collar.copy()
            lithology_df = self.plotter.get_lithology.copy()
            group_df = lithology_df.groupby('borehole_id')
            
            for j, row in collar_df.iterrows():
                top = row['Z']
                bott = group_df.get_group(row['borehole_id']).end_elevation.min()
                ax.plot([row['Y'], row['Y']],[top, bott], '-', c='tab:red')
                ax.annotate(row['borehole_id'], (row['Y'], row['Z']), fontsize=12, ha='right')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title(f'Slice {i}')
        return
    
    def report_voxel_volume(self):
        """
        Compares the voxel lithology volume portions against the sounding 
        lithology volume/linear feet portions
        """
        # Get the lithology type and data
        plot_type = self.plotter.plot_type
        df = self.plotter.lithology_df.copy()
        
        # Calculate the height of each interval
        df['sound_height'] = df['start_elevation'] - df['end_elevation']
        
        # Calculate the total height for each lith type
        counts_df = df.groupby(plot_type)['sound_height'].sum().reset_index()
        
        # Calculate the proportion of each SBT type
        total_height = counts_df['sound_height'].sum()
        counts_df['sound_volume_perc'] = (counts_df['sound_height'] / total_height) * 100
        
        # Sort for better readability
        counts_df.sort_values(plot_type, inplace=True)
        
        # Compute the voxel representative volume
        perc = []
        points = len(self.grid_values.flatten())
        for index in counts_df[plot_type]:
            if plot_type == 'SBT':
                hits = sum((self.grid_values+1 == index).flatten())
            else:
                hits = sum((self.grid_values == index).flatten())
            perc.append(100 * hits/points)
        counts_df['voxel_volume_perc'] = perc
        print(counts_df)
        
        self.counts = counts_df
        return counts_df
    
    @staticmethod
    def resample_lithology_intervals(interval, lithology_df):
        """
        Resample/averages the sounding lithology on a coarser interval (faster vox/plotting)
        
        IN PROGRESS
        """
        
        # lithology_df.columns = ['borehole_id','depth', 'elevation', 'penetration_resistance', 'friction_ratio', 'SBT_index']
        # 'start_elevation','end_elevation'
        
        resampled_data = []
    
        # Process each borehole separately
        for borehole_id in lithology_df['borehole_id'].unique():
            borehole_data = lithology_df[lithology_df['borehole_id'] == borehole_id]
            borehole_data = borehole_data.sort_values(by='start_depth').reset_index(drop=True)
            
            start_depth = borehole_data['start_depth'].min()
            end_depth = borehole_data['end_depth'].max()
            
            # Create new depth intervals
            new_depths = np.arange(start_depth, end_depth, interval)
            
            for new_depth in new_depths:
                # Find the last known value at the current depth
                previous_data = borehole_data[borehole_data['start_depth'] <= new_depth]
                if not previous_data.empty:
                    last_known = previous_data.iloc[-1]
                    resampled_data.append({
                        'borehole_id': borehole_id,
                        'start_depth': new_depth,
                        'end_depth': new_depth + interval,
                        'start_elevation': last_known['start_elevation'] - (new_depth - last_known['start_depth']),
                        'end_elevation': last_known['start_elevation'] - (new_depth - last_known['start_depth']) - interval,
                        'lithology': last_known['lithology']
                    })
        
        return pd.DataFrame(resampled_data)
    
    