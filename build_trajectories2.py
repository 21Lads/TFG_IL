import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BuildTrajectories:
    """define class to build trajectories' segments from vehicle data"""

    def __init__(self, file_path, output_path):
        #r"C:\Users\omkar\Desktop\TFG\TFG_extra\dataset.csv"
        self.columns = ['Time(s)', 'Vehicle_ID', 'x', 'y', 'v', 'dx', 'dy', 'distance', 'dt',
            'computed_v', 'acceleration', 'heading', 'front_vehicle_x',
            'front_vehicle_v', 'space_headway', 'time_headway',
            'longitudinal_acceleration', 'lateral_acceleration',
            'longitudinal_jerk']
        self.df = pd.read_csv(file_path, sep=",", names=self.columns, comment="#", engine="python")
        self.df = self.df.sort_values(["Vehicle_ID", "Time(s)"]).reset_index(drop=True)

    def find_bounds(self, counts, tolerance):
        """ Find lower and upper bounds around a target value with given tolerance """
        target = counts.median() # target value is median
        lower_bound = target - tolerance
        upper_bound = target + tolerance

        return lower_bound, upper_bound


    def select_vehicles(self):
        """ Select vehicles with average (close to median) frame counts """

        # 0. Choose vehicles that change lanes (have y data) so that change in y > 2 m at some point
        self.df = self.to_numeric(self.df)
        lane_changers = []
        for vid, group in self.df.groupby("Vehicle_ID"):
            if group["y"].max() - group["y"].min() > 2.0:
                lane_changers.append(vid)
        
        #lane_changers = self.df[(self.df["y"].max() - self.df["y"].min()) >= 2.0]["Vehicle_ID"].unique()
        lc_df = self.df[self.df["Vehicle_ID"].isin(lane_changers)].copy()

        # 1. Frame counts per vehicle
        vehicle_counts = lc_df["Vehicle_ID"].value_counts().sort_index()
        
        # 2. Determine bounds around median
        tolerance = 5  # frames
        lower_bound, upper_bound = self.find_bounds(vehicle_counts, tolerance)

        #candidate_counts = vehicle_counts[(vehicle_counts >= lower_bound) & (vehicle_counts <= upper_bound)]
        candidate_ids = vehicle_counts[(vehicle_counts >= lower_bound) & (vehicle_counts <= upper_bound)].index ## gets only ids, not counts

        # 3. If more than 20 vehicles qualify, pick 20 random ones for diversity
        if len(candidate_ids) > 20:
            selected_ids = np.random.choice(candidate_ids, size=20, replace=False)
        else:
            selected_ids = candidate_ids[:20]

        #print(f"Selected {len(selected_ids)} vehicles near average frame count:") #Final selected ids
        #print("The selected IDs are the following:\n")
        #print(selected_ids)

        #for i in selected_ids:
        #    print(vehicle_counts[i])

        return selected_ids

    # Select vehicle IDs of lane changers with the most significant lateral movement (y)
    def select_vehicles_lane_changers(self):
        """ Select vehicles with the highest cumulative lateral movement (lane changers) """
        self.df = self.to_numeric(self.df)
        
        # .diff() calculates the frame-by-frame change in Y.
        # .abs() makes it positive.
        # .sum() adds up all lateral movement over the vehicle's lifetime.
        lat_movement = self.df.groupby("Vehicle_ID")["y"].apply(lambda g: g.diff().abs().sum())
        
        # Filter out vehicles with too few frames (e.g., ghost vehicles or bad data)
        # We still want vehicles that existed long enough to demonstrate a full trajectory
        vehicle_counts = self.df["Vehicle_ID"].value_counts()
        valid_min_frames = 600 # Adjust this based on your dt (e.g., 40 frames = 20 seconds at dt=0.5)
        valid_vehicles = vehicle_counts[vehicle_counts >= valid_min_frames].index
        
        # Keep only the valid vehicles in our lateral movement series
        lat_movement = lat_movement.loc[lat_movement.index.isin(valid_vehicles)]
        
        lat_movement = lat_movement.sort_values(ascending=False)
        
        selected_ids = lat_movement.head(20).index.values
        
        print(f"Selected the top {len(selected_ids)} highest lane changers.")
        # print("Total lateral movement for top 5:\n", lat_movement.head(5))

        return selected_ids
    
    def select_driving_styles(self):
        """ Selects 20 Aggressive and 20 Conservative vehicles based on lifetime kinematics """
        self.df = self.to_numeric(self.df)
        
        # Ignorar los time_headways de 0 (cuando no hay coche delante)
        veh_stats = self.df.groupby("Vehicle_ID").apply(lambda g: pd.Series({
            "mean_v": g["v"].mean(),
            "mean_th": g[g["time_headway"] > 0]["time_headway"].mean(),
            "mean_jerk": g["longitudinal_jerk"].abs().mean()
        })).dropna()

        
        # Agresivos: 30% más rápido Y 30% más pegados al coche de delante
        agg_v_thresh = veh_stats["mean_v"].quantile(0.70)
        agg_th_thresh = veh_stats["mean_th"].quantile(0.30)
        
        # Conservadores: 40% más lentos Y 40% más lejos del coche de delante
        cons_v_thresh = veh_stats["mean_v"].quantile(0.40)
        cons_th_thresh = veh_stats["mean_th"].quantile(0.60)

      
        aggressive_pool = veh_stats[
            (veh_stats["mean_v"] > agg_v_thresh) & 
            (veh_stats["mean_th"] < agg_th_thresh)
        ].index.values

        conservative_pool = veh_stats[
            (veh_stats["mean_v"] < cons_v_thresh) & 
            (veh_stats["mean_th"] > cons_th_thresh)
        ].index.values

        
        np.random.seed(42) # Para reproducibilidad en la tesis
        selected_aggressive = np.random.choice(aggressive_pool, size=min(20, len(aggressive_pool)), replace=False)
        selected_conservative = np.random.choice(conservative_pool, size=min(20, len(conservative_pool)), replace=False)

        print(f"Extracted {len(selected_aggressive)} Aggressive and {len(selected_conservative)} Conservative vehicles.")
        return selected_aggressive, selected_conservative
    
    def build_trajectories(self, selected_ids):
        # 1. Extract data for selected vehicles
        selected_df = self.df[self.df["Vehicle_ID"].isin(selected_ids)].copy()
        window = 100   # frames per trajectory
        step = 50      # overlap between trajectories
        max_per_vehicle = 8  # number of trajectories per vehicle
        #Trajectory 1 → Frames [0 : window]
        #Trajectory 2 → Frames [step : step + window]
        #Trajectory 3 → Frames [2*step : 2*step + window]
        #...
        trajectory_segments = []
        traj_counter = 0

        for vid in selected_ids:
            veh = selected_df[selected_df["Vehicle_ID"] == vid].sort_values("Time(s)").reset_index(drop=True)
            local_count = 0

            for start in range(0, len(veh) - window, step):
                if local_count >= max_per_vehicle:
                    break  # stop once 8 trajectories per vehicle are created
                
                seg = veh.iloc[start:start + window].copy()
                seg["traj_id"] = f"{vid}_{local_count}"
                seg["entries_count"] = len(seg)
                trajectory_segments.append(seg)

                local_count += 1
                traj_counter += 1

        print(f"\nTotal trajectories created: {traj_counter}")

        trajectories_df = pd.concat(trajectory_segments, ignore_index=True)

        trajectories_df = self.to_numeric(trajectories_df)

        trajectories_df = trajectories_df.sort_values(by=["traj_id", "Time(s)"]).reset_index(drop=True)

        return trajectories_df
    

    def check_trajectories(self, trajectories_df, length):
        """ ensure 8 trajectories per vehicle"""

        check = trajectories_df.groupby("traj_id")["Vehicle_ID"].first().value_counts()
        #print("\nTrajectories per Vehicle:")
        #print(check)

        for vid in check.index:
            if check[vid] != length:
                return False
            
        return True
    
    def to_numeric(self, df):
        """Convert specified columns to numeric, coercing errors to NaN"""
        # 8. Make trajectories_df all numeric

        # Convert all numeric columns
        df[self.columns] = df[self.columns].apply(pd.to_numeric, errors='coerce')
        #for col in numeric_cols:
        #    if col in df.columns:
        #        df[col] = pd.to_numeric(df[col], errors='coerce')

        # fill NaNs with 0 
        df = df.fillna(0)

        return df

    def save_data(self, trajectories_df, output_path):
        """Save trajectories DataFrame to CSV"""

        trajectories_df.to_csv(output_path, index=False)
        print(f"\nSaved trajectories to {output_path}\n")


    def run(self, output_path):
        """ Main execution method """
        selected_ids = self.select_vehicles()
        #aggressive_ids, conservative_ids = self.select_driving_styles()
        #lane_changer_ids = self.select_vehicles_lane_changers()

        # select 20 vehicles from any subset of the above (e.g., aggressive_ids) to build trajectories
        trajectories_df = self.build_trajectories(selected_ids)

        valid = self.check_trajectories(trajectories_df, length=8)
        if valid:
            print("\nAll selected vehicles have 8 trajectories each.\n")
        else:
            print("\Alert!: Some vehicles do not have 8 trajectories.\n")

        self.save_data(trajectories_df, output_path)
        #="M40_d07_h08_trajectories_updated.csv"

        return trajectories_df
 



#Sort trajectories_df by traj_id and Time(s)
#print(trajectories_df.iloc[310:360])
#print(trajectories_df.dtypes)



# 9. Metadata

#metadata = []
#
#for traj_id, group in trajectories_df.groupby("traj_id"):
#    vid = group["Vehicle_ID"].iloc[0]
#
#    # Duration and displacement
#    duration = group["Time(s)"].max() - group["Time(s)"].min()
#    total_displacement = np.nansum(group["distance"])
#
#    # Lane (y)
#    y_pos = group["y"].mean()
#    lane = int((y_pos % 10) // 2 + 1) if y_pos != 6 else 3 # lane width of 2 meters
#
#    # Velocity stats
#    v_mean = group["v"].mean()
#
#    # Longitudinal and lateral acceleration
#    acc_long_mean = group["longitudinal_acceleration"].mean()
#    acc_lat_mean = group["lateral_acceleration"].mean()
#
#    # Longitudinal jerk (smoothness)
#    jerk_long_mean = group["longitudinal_jerk"].mean()
#
#    # Headways (risk-related)
#    space_headway_mean = group["space_headway"].mean()
#    time_headway_mean = group["time_headway"].mean()
#
#    # Heading change
#    heading_change = group["heading"].iloc[-1] - group["heading"].iloc[0]
#
#    # Derived metrics
#    #efficiency_index = total_displacement / duration if duration > 0 else np.nan
#    #comfort_index = 1 / (1 + abs(jerk_long_mean) + abs(acc_lat_mean))
#
#    # Save all metrics
#    metadata.append({
#        "traj_id": traj_id,
#        "Vehicle_ID": vid,
#        "entries_count": len(group),
#        "duration_s": duration,
#        "displacement_m": total_displacement,
#        "lane": lane,
#        "v_mean": v_mean,
#        "acc_long_mean": acc_long_mean,
#        "acc_lat_mean": acc_lat_mean,
#        "jerk_long_mean": jerk_long_mean,
#        "space_headway_mean": space_headway_mean,
#        "time_headway_mean": time_headway_mean,
#        "heading_change": heading_change,
#        #"efficiency_index": efficiency_index,
#        #"comfort_index": comfort_index
#    })
#
## Convert to DataFrame
#metadata_df = pd.DataFrame(metadata)
#
## === Save metadata to file ===
#output_meta_path = "M40_d07_h08_trajectory_metadata_updated.csv"
##metadata_df.to_csv(output_meta_path, index=False)
#
#print(f"Saved {len(metadata_df)} trajectories' metadata to {output_meta_path}\n")
##print(metadata_df["displacement_m"].head(24))
##print(metadata_df["v_mean"].head(24))
##print(metadata_df["acc_long_mean"].head(24))
##print(metadata_df["acc_lat_mean"].head(24))
##print(metadata_df["jerk_long_mean"].head(24))
##print(metadata_df["space_headway_mean"].head(24))
##print(metadata_df["lane"].head(24))
#
#for i in range(1):#len(selected_ids)):
#    sample_vid = selected_ids[i]
#    sample_traj = metadata_df[metadata_df["Vehicle_ID"] == sample_vid]
#    # Now make a graph of this sample_traj
#    #print(sample_traj)
#    #plt.figure(figsize=(10,6))
#    #plt.plot(sample_traj["Time(s)"], sample_traj["x"], marker='o', linestyle='-', color='blue')
#    #plt.show()
#
#    # plot graph for lane vs time for a specific vehicle
#    plt.figure(figsize=(10,6))
#    plt.plot(sample_traj["duration_s"], sample_traj["lane"], marker='o', linestyle='-', color='green')
#    plt.title(f"Vehicle {sample_vid} Lane Position Over Time")
#    plt.xlabel("Time (s)")
#    plt.ylabel("Lane Position (y)")
#    plt.show()

