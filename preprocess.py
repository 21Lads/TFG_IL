import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.preprocessing import StandardScaler

class Preprocess:
    """Class to preprocess raw vehicle trajectory data"""

    def __init__(self, file_path, output_path):
        #r"C:\Users\omkar\Desktop\TFG\TFG_extra\M40-d07-h08.dat"
        self.columns = ["Time(s)", "Vehicle_ID", "x", "y", "v"]
        self.df = pd.read_csv(file_path, sep=r"\s+", names=self.columns, comment='#', engine='python')  
        self.df = self.df.sort_values(by=["Vehicle_ID", "Time(s)"]).reset_index(drop=True)
        self.df = self.clean_data(self.df)
        self.run(output_path)
        #self.plot_outliers(self.df, self.columns)


    def calculate_unclean(self, df_clean):
        """ Calculate and print percentage of unclean data """
        unclean = df_clean.isnull().sum(axis = 0)
        df_clean = df_clean.dropna().reset_index(drop=True)
        unclean_after = df_clean.isnull().sum(axis = 0)
        # Calculate percentage of unclean data cells removed
        res = ((unclean.sum() - unclean_after.sum()) / len(df_clean)) * 100

        return res, df_clean

    def clean_data(self, df_cleaned):
        """ Clean data by removing duplicates and interpolating missing values """

        # Convert columns to numeric (if they aren't already)
        df_cleaned['x'] = pd.to_numeric(df_cleaned['x'], errors='coerce')
        df_cleaned['y'] = pd.to_numeric(df_cleaned['y'], errors='coerce')

        # Fill missing positions (x, y) using cubic interpolation per vehicle
        df_cleaned['x'] = df_cleaned.groupby('Vehicle_ID')['x'].transform(lambda group: group.interpolate(method='cubic').ffill()
         .bfill())
        df_cleaned['y'] = df_cleaned.groupby('Vehicle_ID')['y'].transform(lambda group: group.interpolate(method='cubic').ffill()
         .bfill())

        # For velocities or other continuous fields, use median imputation if skewed
        #df_cleaned['v'] = df_cleaned['v'].fillna(df_cleaned['v'].median())

        # for v use linear interpolation since it's more likely to be smooth and less skewed only for null values
        df_cleaned['v'] = df_cleaned.groupby('Vehicle_ID')['v'].transform(lambda group: group.interpolate(method='linear').ffill()
         .bfill())

        res, df_cleaned = self.calculate_unclean(df_cleaned)
        print(f"Initial unclean data percentage: {res}%")

        return df_cleaned
    
# Now the data is ready for modeling or further analysis

    def motion_features(self, df):
        """Compute motion features """

        df["dx"] = df.groupby("Vehicle_ID")["x"].diff()
        df["dy"] = df.groupby("Vehicle_ID")["y"].diff()
        df["distance"] = np.sqrt(df["dx"]**2 + df["dy"]**2) # Euclidean distance since 2d coordinates only
        df["dt"] = df.groupby("Vehicle_ID")["Time(s)"].diff() # Always 500ms but still

        #df["computed_v"] = df["distance"] / df["dt"]                      # computed velocity in m/s
        df["computed_v"] = (df["v"] * 1000) / 3600                          # computed velocity in m/s
        #df["acceleration"] = df.groupby("Vehicle_ID")["v"].diff() / df["dt"]  # acceleration with existing velocity in km/s^2
        df["acceleration"] = df.groupby("Vehicle_ID")["computed_v"].diff().div(df["dt"])  # acceleration with existing velocity in m/s^2
        df["heading"] = np.arctan2(df["dy"], df["dx"])                    # direction angle

        res, df = self.calculate_unclean(df)
        print(f"Unclean data percentage after motion features: {res}%")

        return df

    def derive_additional_features(self, df_clean):

        # Sort by time and position
        df_clean = df_clean.sort_values(by=["Time(s)", "y", "x"]).reset_index(drop=True)

        # Compute nearest front vehicle at each timestamp 
        # Basically, find for each time instant the next vehicle ahead (bigger x value)
        df_clean["front_vehicle_x"] = df_clean.groupby(["Time(s)", "y"])["x"].shift(-1) # move all rows one above since head car doesn't have a front vehicle
        df_clean["front_vehicle_v"] = df_clean.groupby(["Time(s)", "y"])["v"].shift(-1)


        ## Compute Space and Time Headway 
        df_clean["space_headway"] = df_clean["front_vehicle_x"] - df_clean["x"] # in meters
        df_clean["time_headway"] = df_clean["space_headway"].div(df_clean["computed_v"]) # in seconds
        df_clean.loc[df_clean["time_headway"] < 0, "time_headway"] = np.nan  # remove invalid (behind) cases


        # Compute longitudinal acceleration, longitudinal jerk, lateral acceleration; both ways of calculation give the same result

        #df_clean["longitudinal_acceleration1"] = df_clean.groupby("Vehicle_ID")["v"].diff() / df_clean["dt"]
        df_clean["longitudinal_acceleration"] = df_clean["acceleration"] * np.cos(df_clean["heading"])

        #df_clean["lateral_acceleration1"] = df_clean.groupby("Vehicle_ID")["heading"].diff() / df_clean["dt"]
        df_clean["lateral_acceleration"] = df_clean["acceleration"] * np.sin(df_clean["heading"])

        df_clean["longitudinal_jerk"] = df_clean.groupby("Vehicle_ID")["longitudinal_acceleration"].diff().div(df_clean["dt"])

        df_clean = df_clean.dropna().reset_index(drop=True)


        res, df_clean = self.calculate_unclean(df_clean)
        print(f"Unclean data percentage after additional features: {res}%")

        return df_clean


    def compute_columns(self):
        self.df = self.motion_features(self.df)
        self.df = self.derive_additional_features(self.df)

    def save_data(self, output_path):
        """ Save cleanly with float format"""

        self.df.to_csv(
            output_path,
            index=False,
            float_format="%.6f"
        )

    def run(self, output_path):
        """ Main execution method """
        self.compute_columns()
        self.save_data(output_path)


    def plot_outliers(self, df, columns):
        """ Plot outliers for specified columns using boxplots and scatterplots """
        for column in columns:
            plt.figure(figsize=(10, 5))

            if column != 'Vehicle_ID':  # Skip categorical-like columns for scatterplots
                # Boxplot for Outliers Detection
                plt.subplot(1, 2, 1)
                sns.boxplot(data=df[column])
                plt.title(f'Boxplot of {column}')

                # Scatterplot to check correlations with other variables (useful for numerical columns)
                plt.subplot(1, 2, 2)
                sns.scatterplot(x=df['Time(s)'], y=df[column])
                plt.title(f'Scatterplot of {column} vs Time(s)')

            plt.tight_layout()
            plt.show()


"""import math as m

sample = df_clean.iloc[10]
longitudinal_acc = m.cos(sample["heading"]) * sample["acceleration"]
lateral_acc = m.sin(sample["heading"]) * sample["acceleration"]

print(f"longitudinal: {longitudinal_acc}\nlateral: {lateral_acc}")
"""
#print(df_clean.dtypes.head(10))