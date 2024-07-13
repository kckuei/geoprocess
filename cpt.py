import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler

from mappings import (
    sbt_mapping, 
    color_mapping
)



def mkdir_if_no_exist(outpath_folder):
    """ 
    Setup output directory if it doesnt exist in current working directory
    """
    out_directory = os.path.join(os.getcwd(),outpath_folder)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    return out_directory

def get_files_with_extension(directory, extension):
    """
    Given a path and file extensions, returns a list of the full paths, and fnames
    of files ending in that extension.
    """
    files =[]
    fnames = []
    for file in os.listdir(directory):
        if file.lower().endswith(extension):
            files.append(os.path.join(directory, file))
            fnames.append(file)
    return files, fnames


def load_collar(path, normalize=True, preview=True, savefig=False, output_folder=""):
    """
    Load and return collar data as dataframe, preview and normalize optional.
    """
    # Get the collar data
    collar_df = pd.read_csv(path)
    
    if normalize:
        # Normalize lat/long plotting
        scaler = MinMaxScaler()
        collar_df[['long', 'lat']] = scaler.fit_transform(collar_df[['long', 'lat']])
    
    if preview:
        # Plot the borehole locations
        fig, ax = plt.subplots()
        plt.plot(collar_df.long, collar_df.lat, 'tab:red', ls='None', 
                 marker=r"$\bigoplus$", ms=10, mew=0.2)
        for i, row in collar_df.iterrows():
            ax.annotate(row['id'], (row['long'], row['lat']), fontsize=12, ha='right')
        txt = ""
        if normalize:
            txt = "Normalized"
        plt.xlabel(f'{txt} Long')
        plt.ylabel(f'{txt} Lat')
        if savefig:
            output_path = mkdir_if_no_exist(output_folder)
            fig.savefig(os.path.join(output_path,"latlong")+".svg", dpi=300, bbox_inches="tight")
        plt.show()
    
    return collar_df

def load_cpt_data(fname, sheetnames):
    """
    Given a fname and sheetnames, returns a dictionary of dataframes wit keys
    correspondingt ot he CPTs, and a merged dataframe of all data with an 
    additional field corresponding to the sounding ID.
    """
    cpt_data = {}
    for sheet in sheetnames:
        # Read the column names
        df_columns = pd.read_excel(fname, "Cols")
        columns = df_columns["Column Name - Short"].tolist()
        
        # Read the data from each sheet
        df = pd.read_excel(fname, sheet, skiprows=27)
        df.columns = columns
        
        # Add an identifier for the source sheet
        key = sheet.replace('Liq_', '')
        df['Sounding'] = key
        
        # Reset the index
        df = df.reset_index(drop=True)
        
        # Add the DataFrame to the dictionary
        cpt_data[key] = df
    
    # Merge all DataFrames into one
    merged_df = pd.concat(cpt_data.values(), ignore_index=True)
    
    return cpt_data, merged_df



def create_intervals(df, field):
    """
    Create a DataFrame with contiguous intervals based on a specified field.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    field (str): The field name to create intervals based on.

    Returns:
    pd.DataFrame: A DataFrame with contiguous intervals.
    """
    intervals = []
    current_value = df[field].iloc[0]
    start_depth = df['depth'].iloc[0]
    start_elevation = df['elev'].iloc[0]

    for i in range(1, len(df)):
        if df[field].iloc[i] != current_value:
            end_depth = df['depth'].iloc[i]
            end_elevation = df['elev'].iloc[i]
            intervals.append((start_depth, end_depth, 
                              start_elevation, end_elevation, 
                              current_value))
            start_depth = end_depth
            start_elevation = end_elevation
            current_value = df[field].iloc[i]

    # Append the last interval
    intervals.append((start_depth, df['depth'].iloc[-1], 
                      start_elevation, df['elev'].iloc[-1], 
                      current_value))
    
    intervals_df = pd.DataFrame(intervals, 
                                columns=['start_depth', 'end_depth', 
                                         'start_elev', 'end_elev', 
                                         field])
    return  intervals_df




def plot_cpt_soundings(cpt_data, plt_soundings, plot_key='cpt-qc', plot_label=None, 
                       color_mapping_layers=None, xlim=None, use_elev=True, savefig=False,
                       output_folder = ""):
    """
    Plot elevation profile comparisons, colored by SBT.
    """
    # Data input validation
    if not all(isinstance(i, list) for i in plt_soundings):
        print("Input must be a nested list of sounding keys.")
        return
   
    unique_keys = np.unique([item for sublist in plt_soundings for item in sublist])
    if not all([i in cpt_data.keys() for i in unique_keys]):
        for i in unique_keys:
            if i not in cpt_data.keys():
                print(f"Key not found in cpt_data: {i}")
                return
    
    # Define axes base width and height
    ax_width = 5
    ax_height = 10
    
    # Plot each of the soundings lists
    for cpt_keys in plt_soundings:
        
        # Setup the figure
        N = len(cpt_keys)
        fig, axes = plt.subplots(ncols=N, figsize=(ax_width*N,ax_height))
        axes = axes.flatten()  
        
        # Filter the parent dict for only the soundings neeeded
        filtered_dict = {key: cpt_data[key] for key in cpt_keys}
        merged_df = pd.concat(filtered_dict.values(), ignore_index=True)
        
        # Get limits for plotting
        y_min = merged_df.elev.min()
        y_max = merged_df.elev.max()   
        x_max = merged_df[plot_key].max()         
        if xlim is not None and isinstance(xlim[1], (int, float)):
            x_max = max(merged_df[plot_key].max(), xlim[1])
    
        # Populate eeach axes with the cpt data
        for ax, key in zip(axes, cpt_keys):
            
            # Get the data
            df = cpt_data[key]
            df = df.dropna(axis=0,how='all')
            
            # Map the text values to numerical indices
            df['sbt_num'] = df['sbt'].map(sbt_mapping)
            
            # Create a new DataFrame with contiguous intervals
            interval_field = 'sbt_num'
            interval_df = create_intervals(df, interval_field)
            
            # Plot the SBT-colored depth intervals
            for _, row in interval_df.iterrows():
                if not np.isnan(row.start_depth):
                    
                    if use_elev:
                        y = [row['start_elev'], row['end_elev']]
                    else:
                        y = [row['start_depth'], row['end_depth']]
                        
                    ax.fill_betweenx(y=y, x1=0, x2=df[plot_key].max(),
                        color=color_mapping[row['sbt_num']],
                        edgecolor='none'
                    )
            
            # Plot the CPT-qc trace
            if use_elev:
                ax.plot(df[plot_key], df['elev'], color='black', linewidth=2)
                ax.fill_betweenx(df['elev'], df[plot_key], x_max, color='white')
            else:
                ax.plot(df[plot_key], df['depth'], color='black', linewidth=2)
                ax.fill_betweenx(df['depth'], df[plot_key], x_max, color='white')
            
            
            # Plot the secondary unit layering if a color mapping is specified
            if color_mapping_layers is not None:
                interval_df2 = create_intervals(df, 'layer')
                
                # Plot the colored depth intervals
                for _, row in interval_df2.iterrows():
                    if not np.isnan(row.start_depth):
                            
                        closest_index_start = (df['elev'] - row['start_elev']).abs().idxmin()
                        closest_index_end   = (df['elev'] - row['end_elev']).abs().idxmin()
            
                        ax.fill_betweenx(df.loc[closest_index_start:closest_index_end,'elev'],
                                         df.loc[closest_index_start:closest_index_end,plot_key],
                                         x_max, color=color_mapping_layers[row['layer']], 
                                         alpha=0.2)

            
            # Invert y-axis so depth increases downwards
            if not use_elev: 
                ax.invert_yaxis()
            
            # Set limits
            if xlim is not None:
                if xlim[0] is not None:
                    ax.set_xlim(left=xlim[0])
                if xlim[1] is not None:
                    ax.set_xlim(right=xlim[1])
            ax.set_ylim(bottom=y_min, top=y_max)
            
            # Set labels and title
            ax.set_xlabel(plot_label if plot_label else plot_key)
            if use_elev:
                ax.set_ylabel('Elevation (ft)')
            else:
                ax.set_ylabel('Depth (ft)')
            ax.set_title(key)
            
            # Gridlines
            ax.grid(which='both', alpha=0.2)
            
        if savefig:
            out_directory = mkdir_if_no_exist(output_folder)
            out_path = os.path.join(out_directory,f"{plot_key}_section_"+"_".join(cpt_keys)+".svg")
            fig.savefig(out_path,dpi=300, bbox_inches="tight")
            
        plt.show()
    return


class CPTPlotter:
    """
    TO BE IMPLEMENTED.
    """


if __name__ == '__main__':
    
    # Get the CPT data
    cpt_files, __ = get_files_with_extension('data\CPTu Data', '.xls')
    cpt_ppd_files, __ = get_files_with_extension('data\CPTu PPD Data', '.xls')
