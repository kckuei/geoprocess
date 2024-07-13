
class CPTData():
    
    def __init__(self, collar_fname, cpt_fname, cpt_sheetnames):

        self.collar_fname = collar_fname
        self.cpt_fname = cpt_fname
        self.cpt_sheetnames = cpt_sheetnames
        
        self.cpt_data, self.merged_df = self.load_cpt_data(cpt_fname, cpt_sheetnames)
        self.soundings = self.cpt_data.keys()
    
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
        return
    
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
    
