import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler

from mappings import (
    sbt_mapping,
    color_mapping
)


class LithologyPlotter:
    
    VALID_PLOT_TYPES = ['SBT', 'penetration_resistance', 'friction_ratio']
    
    def __init__(self, cpt_data, collar_file, plot_type='penetration_resistance', 
                 fric_fac=0.3, pen_fac=0.2, elev_3d=30, azim_3d=60, show_plane=True,
                 radius_3d = 0.10, point1=(0,0), point2=(1,1),
                 savefig=True):
        
        self.validate_inputs(cpt_data, plot_type, fric_fac, pen_fac, collar_file)
        
        self.cpt_data = cpt_data
        self.sbt_mapping = sbt_mapping
        self.plot_type = plot_type
        self.fric_fac = fric_fac
        self.pen_fac = pen_fac
        self.collar_file = collar_file
        
        self.elev_3d = elev_3d
        self.azim_3d = azim_3d
        self.show_plane = show_plane
        self.radius_3d = radius_3d
        self.point1 = point1
        self.point2 = point2

        self.SBT_colors = LinearSegmentedColormap.from_list("", list(color_mapping.values()), N=9)
        self.penetration_resistance_colors = plt.cm.viridis
        self.friction_ratio_colors = plt.cm.inferno
        self.savefig = savefig
        
        self.lithology_df = None
        self.collar_df = None

        self.update()

    def validate_inputs(self, cpt_data, plot_type, fric_fac, pen_fac, collar_file):
        if not isinstance(cpt_data, dict) or not all(isinstance(df, pd.DataFrame) for df in cpt_data.values()):
            raise ValueError("cpt_data must be a dictionary with pandas DataFrame values.")
        
        if plot_type not in self.VALID_PLOT_TYPES:
            raise ValueError(f"plot_type must be one of {self.VALID_PLOT_TYPES}.")
        
        if not (0 <= fric_fac <= 1):
            raise ValueError("fric_fac must be between 0 and 1.")
        
        if not (0 <= pen_fac <= 1):
            raise ValueError("pen_fac must be between 0 and 1.")
        
        if not isinstance(collar_file, str):
            raise ValueError("collar_file must be a string representing the file path.")

        return
    

    @property
    def get_lithology(self):
        return self.lithology_df
    
    @property    
    def get_collar(self):
        return self.collar_df
    
    def set_view(self, elev_3d, azim_3d):
        self.elev_3d, self.azim_3d = elev_3d, azim_3d
        return
    
    def set_projection_line(self, point1, point2):
        self.point1, self.point2 = point1, point2
        return

    def update(self):
        """
        Update lithology
        """
        self.lithology_df, self.collar_df = self.load_and_prepare_data()
        self.lithology_df, self.cmap_mappable_arr = self.cap_values()
        
        if self.plot_type == 'SBT':
            self.lithology_df, _ = self.combine_segments()
            self.lithology_colors = self.SBT_colors
        else:
            self.lithology_df, bins = self.combine_segments(property_name=self.plot_type, n_bins=10)
            self.lithology_colors = self.penetration_resistance_colors if self.plot_type == 'penetration_resistance' else self.friction_ratio_colors
            self.lithology_colors = self.lithology_colors(np.linspace(0, 1, 10))
        return

    def load_and_prepare_data(self):
        """
        Load and prepare the data for plotting
        """
        merged_df = pd.concat(self.cpt_data.values(), ignore_index=True)
        merged_df['sbt_num'] = merged_df['sbt'].map(self.sbt_mapping)
        
        lithology_df = merged_df[['Sounding','depth', 'elev', 'cpt-qc','F','sbt_num']].copy()
        # Rename the columns
        lithology_df.columns = ['borehole_id','depth', 'elevation', 'penetration_resistance', 'friction_ratio', 'SBT']
        # Sort the data
        lithology_df.sort_values(['borehole_id','depth'], inplace=True)
        
        collar_df = pd.read_csv(self.collar_file)
        # Normalize the collar data
        scaler = MinMaxScaler()
        collar_df[['long', 'lat']] = scaler.fit_transform(collar_df[['long', 'lat']])
        # Rename the columns
        collar_df.columns = ['borehole_id', 'X', 'Y', 'Z', 'gw-depth']
        
        return lithology_df, collar_df

    def cap_values(self):
        """
        Function for capping friction ratio or penetration resistance and returning
        mapping arrays for colormaps
        """
        if self.plot_type == 'friction_ratio':
            max_fric = self.lithology_df.friction_ratio.max()
            cap = max_fric * min(max(0, self.fric_fac), 1)
            self.lithology_df.loc[self.lithology_df.friction_ratio > cap, 'friction_ratio'] = cap
            cmap_mappable_arr = self.lithology_df['friction_ratio'].copy()
        elif self.plot_type == 'penetration_resistance':
            max_pen = self.lithology_df.penetration_resistance.max()
            cap = max_pen * min(max(0, self.pen_fac), 1)
            self.lithology_df.loc[self.lithology_df.penetration_resistance > cap, 'penetration_resistance'] = cap
            cmap_mappable_arr = self.lithology_df['penetration_resistance'].copy()
        else:
            cmap_mappable_arr = None
        return self.lithology_df, cmap_mappable_arr

    def combine_segments(self, property_name=None, n_bins=10):
        """
        Function for combining continugous intervals of the same type/range
        """
        combined_data = []
        for borehole_id in self.lithology_df['borehole_id'].unique():
            sounding_data = self.lithology_df[self.lithology_df['borehole_id'] == borehole_id].sort_values('depth').reset_index(drop=True)
            
            if property_name:
                min_value = self.lithology_df[property_name].min()
                max_value = self.lithology_df[property_name].max()
                bins = np.linspace(min_value, max_value, n_bins + 1)
                categories = np.digitize(sounding_data[property_name], bins) - 1
                categories[categories == n_bins] = n_bins - 1  # Fix any out-of-range indices
            else:
                categories = sounding_data['SBT']
            
            start_idx = 0
            for i in range(1, len(sounding_data)):
                if categories[i] != categories[start_idx]:
                    combined_data.append({
                        'borehole_id': borehole_id,
                        'start_elevation': sounding_data['elevation'][start_idx],
                        'end_elevation': sounding_data['elevation'][i],
                        self.plot_type: categories[start_idx]
                    })
                    start_idx = i
            
            # Add the last segment
            combined_data.append({
                'borehole_id': borehole_id,
                'start_elevation': sounding_data['elevation'][start_idx],
                'end_elevation': sounding_data['elevation'][len(sounding_data)-1],
                self.plot_type: categories[start_idx]
            })
        
        combined_df = pd.DataFrame(combined_data)
        return combined_df, bins if property_name else None

    def create_cylinder_data(self, x, y, start_elevation, end_elevation, radius=0.05, num_sides=20):
        """
        Function to create cylinders
        """
        z = [start_elevation, end_elevation]
        theta = np.linspace(0, 2*np.pi, num_sides)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + x
        y_grid = radius * np.sin(theta_grid) + y
        return x_grid, y_grid, z_grid

    def project_onto_plane(self, x, y, point1, point2):
        """
        Function to project soundings onto verticla plane
        """
        line_vec = np.array(point2) - np.array(point1)
        point_vec = np.array([x, y]) - np.array(point1)
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        projection_length = np.dot(point_vec, line_unitvec)
        projection = point1 + projection_length * line_unitvec
        return projection[0], projection[1]

    def create_3d_plot(self, ax1):
        """
        Creates a 3D lithology plot of soundings
        """
        for bh in self.collar_df['borehole_id']:
            collar = self.collar_df[self.collar_df['borehole_id'] == bh]
            lithologies = self.lithology_df[self.lithology_df['borehole_id'] == bh]
            
            for i in range(0, len(lithologies)):
                start_elevation = lithologies.iloc[i]['start_elevation']
                end_elevation = lithologies.iloc[i]['end_elevation']
                lithology = lithologies.iloc[i][self.plot_type]
                
                x = collar['X'].values[0]
                y = collar['Y'].values[0]
                
                X, Y, Z = self.create_cylinder_data(x, y, start_elevation, end_elevation, radius=self.radius_3d)
                
                if self.plot_type == 'SBT':
                    color = self.SBT_colors(lithology - 1)  # Adjust for zero-based index
                else:
                    color = self.lithology_colors[lithology]
                
                ax1.plot_surface(X, Y, Z, color=color, alpha=0.75)
                
            ax1.scatter([x], [y], [max(lithologies.start_elevation)], color='k', marker='o', s=50, zorder=1e6)
            ax1.text(x, y, collar['Z'].values[0], bh, color='black', zorder=1e6)

        ax1.set_xlabel('X (Longitude)')
        ax1.set_ylabel('Y (Latitude)')
        ax1.set_zlabel('Elevation (ft, NGVD29)')
        ax1.view_init(elev=self.elev_3d, azim=self.azim_3d)

        if self.show_plane:
            self.draw_projection_plane(ax1)
        return

    def draw_projection_plane(self, ax1, zbuffer = 10):
        """
        Draws projection plane on 3d axes
        """
        x_plane = np.array([self.point1[0], self.point2[0]])
        y_plane = np.array([self.point1[1], self.point2[1]])
        z_plane = np.linspace(self.lithology_df['end_elevation'].min() - zbuffer, 
                              self.lithology_df['start_elevation'].max() + zbuffer, 10)
        
        X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
        Y_plane = np.tile(y_plane, (len(z_plane), 1))

        ax1.plot_surface(X_plane, Y_plane, Z_plane, color='lightblue', alpha=0.2)

        x_line = np.linspace(self.point1[0], self.point2[0], 100)
        y_line = np.linspace(self.point1[1], self.point2[1], 100)
        z_line = np.full_like(x_line, self.lithology_df['end_elevation'].min() - zbuffer)
        ax1.plot(x_line, y_line, z_line, c='red', linewidth=2)

        ax1.scatter([self.point1[0]], [self.point1[1]], [self.lithology_df['end_elevation'].min() - zbuffer], facecolors='none', edgecolors='r', s=100, marker='o', linewidths=2)
        ax1.scatter([self.point2[0]], [self.point2[1]], [self.lithology_df['end_elevation'].min() - zbuffer], facecolors='none', edgecolors='r', s=100, marker='^', linewidths=2)
        return
    
    def create_2d_profile_plot(self, ax2):
        """
        Draws borehole lithology on 2d projection
        """
        for bh in self.collar_df['borehole_id']:
            collar = self.collar_df[self.collar_df['borehole_id'] == bh]
            lithologies = self.lithology_df[self.lithology_df['borehole_id'] == bh]
            
            x_proj, y_proj = self.project_onto_plane(collar['X'].values[0], collar['Y'].values[0], self.point1, self.point2)
            
            for i in range(0, len(lithologies)):
                start_elevation = lithologies.iloc[i]['start_elevation']
                end_elevation = lithologies.iloc[i]['end_elevation']
                lithology = lithologies.iloc[i][self.plot_type]
                
                if self.plot_type == 'SBT':
                    color = self.SBT_colors(lithology - 1)  # Adjust for zero-based index
                else:
                    color = self.lithology_colors[lithology]
                
                if True:
                    # Draw as line segments (faster)
                    bhwidth=12
                    ax2.plot([y_proj, y_proj], [start_elevation, end_elevation], 
                             color=color, linewidth=bhwidth)
                else:
                    # Draw as areas
                    bhwidth = 0.01
                    ax2.fill_between([y_proj-bhwidth, y_proj+bhwidth],
                                      start_elevation, end_elevation, 
                                      color=color, edgecolor='black')
            
            bhtop = lithologies.start_elevation.max()
            voffset = 0.5
            ax2.text(y_proj, bhtop + voffset, bh, color='black', verticalalignment='bottom')

        ax2.set_ylim(ax2.get_ylim())
        ax2.set_xlim(ax2.get_xlim())
        ax2.plot(ax2.get_xlim(), [ax2.get_ylim()[0]]*2, 'r', lw=3, clip_on=False)
        ax2.plot(ax2.get_xlim()[0], ax2.get_ylim()[0], 'ro', mfc='w', mew=3, ms=15, clip_on=False)
        ax2.plot(ax2.get_xlim()[1], ax2.get_ylim()[0], 'r^', mfc='w', mew=3, ms=15, clip_on=False)

        ax2.set_xlabel('Projected Distance')
        ax2.set_ylabel('Elevation (ft)')
        ax2.grid(which='both', alpha=0.4)
        return

    def add_colorbar(self, fig, ax1):
        """
        Adds a colorbar to axes
        """
        if self.plot_type == 'SBT':
            handles = [plt.Line2D([0], [0], color=self.SBT_colors(i), lw=10) for i in range(9)]
            labels = list(self.sbt_mapping.keys())
            ax1.legend(handles, labels, ncol=3, loc='best', title="Soil Behavior Type" if self.plot_type == 'SBT' else self.plot_type.replace('_', ' ').title())

        elif self.plot_type != 'SBT':
            mappable = plt.cm.ScalarMappable(cmap=self.penetration_resistance_colors if self.plot_type == 'penetration_resistance' else self.friction_ratio_colors)
            mappable.set_array(self.cmap_mappable_arr)
            cbar = fig.colorbar(mappable, ax=ax1, orientation='horizontal', location='bottom', fraction=0.02, shrink=0.8, pad=0.0)
            cbar.set_label(self.plot_type.replace('_', ' ').title())
            
            if (self.fric_fac < 1.0 and self.plot_type == 'friction_ratio') or (self.pen_fac < 1.0 and self.plot_type == 'penetration_resistance'):
                ticks = cbar.get_ticks()
                tick_labels = [str(int(tick)) for tick in ticks]
                tick_labels[-1] = f'>{int(ticks[-1])}'
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(tick_labels)      
        return

    def plot(self, exts=['.svg']):
        """
        Main lithology plotting function
        """
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        self.create_3d_plot(ax1)
        self.create_2d_profile_plot(ax2)
        self.add_colorbar(fig, ax1)
        plt.tight_layout()

        if self.savefig:
            for ext in exts:
                fig.savefig(f'lithology_{self.plot_type}'+ext, dpi=300, bbox_inches="tight")

        plt.show()
        return

