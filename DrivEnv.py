import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance

class DrivingEnv:
    """
    A spatial driving environment for MaxEnt IRL.
    The environment simulates vehicle kinematics based on agent actions,
    using the dataset strictly for initialization and expert demonstrations.
    """
        
    def __init__(self, trajectory_data: pd.DataFrame, step_size=1, lane_width=2.0, road_width=6, dt=0.5):
        self.data = trajectory_data
        self.traj_ids = trajectory_data["traj_id"].unique()
        self.step_size = step_size
        self.lane_width = lane_width
        self.road_width = road_width
        self.lanes = int(road_width / lane_width)
        self.dt = dt  # Time step duration (0.5 seconds per frame)
        
        self.current_index = 0 
        self.current_traj = None 
        self.current_traj_idx = 0 
        self.done = False
        
        # Agent's kinematic state
        self.x_pos = 0.0 
        self.y_pos = 0.0 
        self.v = 0.0 
        self.accel = 0.0 # Keep track of acceleration for features
        self.jerk = 0.0  # Keep track of jerk for features
        self.prev_y_pos = 0.0 # To calculate lateral acceleration
        self.prev_vy = 0.0 # To calculate lateral acceleration

    def _next_traj_id(self):
        if self.current_traj_idx >= len(self.traj_ids):
            print("No more trajectories left.")
            return None
        traj = self.traj_ids[self.current_traj_idx]
        self.current_traj_idx += 1
        return traj

    def reset(self, traj_id=None):
        """Reset the environment to the start of a specific trajectory."""
        if traj_id is None:
            traj_id = self._next_traj_id()
            if traj_id is None:
                raise RuntimeError("No more trajectories available to reset.")

        self.current_traj = self.data[self.data["traj_id"] == traj_id].reset_index(drop=True)
        self.current_index = 0
        self.done = False

        # Initialize the agent EXACTLY where the expert started
        self.x_pos = float(self.current_traj["x"].iloc[0])
        self.y_pos = float(self.current_traj["y"].iloc[0]) if "y" in self.current_traj.columns else 500.0
        self.v = float(self.current_traj["computed_v"].iloc[0])
        self.accel = 0.0
        self.jerk = 0.0
        self.prev_y_pos = 0.0 # To calculate lateral acceleration
        self.prev_vy = 0.0 # To calculate lateral acceleration
        
        return self.get_state()

    def get_state(self):
        """Return the current state features for the IRL agent (7 Features)."""
        if self.current_traj is None or self.current_index >= len(self.current_traj):
            return None
            
        features = self._features()
        return np.array([
            features["longitudinal_speed"],
            features["ego_longitudinal_acc"],
            features["ego_lateral_acc"],
            features["ego_longitudinal_jerk"],
            features["front_risk"],
            features["collision"],
            features["interaction"]
        ], dtype=np.float32)

    def _features(self):
        """
        Extract the 7 features based on the AGENT'S current kinematics 
        and the environment's traffic state.
        """
        dataset_row = self.current_traj.iloc[self.current_index]
        
        ego_longitudinal_speed = self.v 
        
        ego_longitudinal_acc = self.accel
        ego_longitudinal_jerk = self.jerk
        # lateral acceleration is approximated
        ego_lateral_acc = 0.0
        if self.prev_y_pos != 0.0:
            # Velocidad lateral (metros / segundo)
            current_vy = (self.y_pos - self.prev_y_pos) / self.dt
            # Aceleración lateral (m / s^2)
            ego_lateral_acc = (current_vy - self.prev_vy) / self.dt
        
        th_front = dataset_row.get("time_headway", 0.0)
        front_risk = np.exp(-th_front) if th_front > 0 else 0.0
        
        distance_front = th_front * self.v if th_front > 0 else 999.0
        collision = 1.0 if distance_front < 2.0 else 0.0
        
        accel_data = dataset_row.get("acceleration", 0.0)
        interaction = 0.0
        if self.accel < 0 and front_risk > 0.05:
            interaction = abs(self.accel)

        return {
            "longitudinal_speed": ego_longitudinal_speed,
            "ego_longitudinal_acc": ego_longitudinal_acc,
            "ego_lateral_acc": ego_lateral_acc,
            "ego_longitudinal_jerk": ego_longitudinal_jerk,
            "front_risk": front_risk,
            "collision": collision,
            "interaction": interaction
        }

    def step(self, action):
        """
        Take one step in the environment using physics/kinematics.
        action = [delta v, delta lane]
        """
        if self.done or self.current_traj is None:
            raise RuntimeError("Environment not reset or already finished.")

        delta_v, delta_lane = action 
        
        # Update Kinematics based on actions
        new_v = max(0, self.v + delta_v) # Prevent negative speeds
        
        # Calculate Accel and Jerk for features before updating
        new_accel = (new_v - self.v) / self.dt
        self.jerk = (new_accel - self.accel) / self.dt
        self.accel = new_accel
        self.v = new_v
        self.prev_y_pos = self.y_pos  # Store previous y for lateral acceleration calculation
        self.prev_vy = (self.y_pos - self.prev_y_pos) / self.dt if self.prev_y_pos != 0.0 else 0.0 # Store previous lateral velocity
        # Update X and Y Positions
        self.x_pos += self.v * self.dt  # Agent drives forward based on its own speed
        self.y_pos += delta_lane * self.lane_width
        
        # CLIP Y POSITION: Prevent agent from driving off the road
        self.y_pos = np.clip(self.y_pos, 500.0, 506.0)
        #self.v = np.clip(self.v, 0.0, 35.0) # Clip speed to a reasonable range

        # Advance Time
        self.current_index += self.step_size
        if self.current_index >= len(self.current_traj):
            self.done = True

        next_state = self.get_state()
        
        # In IRL, environment reward should ideally be 0, because the agent 
        # is supposed to calculate reward using theta * features. 
        # Returning 0 prevents interference.
        reward = 0.0 
        
        info = {"x": self.x_pos, "y": self.y_pos, "velocity": self.v}

        return next_state, reward, self.done, info

    def simulate_expert(self, traj_id):
        """
        Simulates the EXPERT trajectory using the dataset actions.
        Useful for extracting demonstrations or visualizing ground truth.
        """
        self.reset(traj_id)
        trajectory_record = []

        for idx in range(1, len(self.current_traj)):
            prev = self.current_traj.iloc[idx - 1]
            curr = self.current_traj.iloc[idx]

            delta_v = curr["v"] - prev["v"]
            delta_y = curr["y"] - prev["y"]
            delta_lane = int(round(delta_y / self.lane_width))

            next_state, _, done, info = self.step([delta_v, delta_lane])

            trajectory_record.append({
                "time_step": idx,
                "x": info["x"],
                "y": info["y"],
                "velocity": info["velocity"],
                "delta_v": delta_v,
                "delta_lane": delta_lane
            })
            
            if done: break

        return pd.DataFrame(trajectory_record)



    def simulate_agent(self, mdp_builder, policy, traj_id, max_steps):
        """
        Simulates the learned policy in the Driving Environment for a specific trajectory.
        """
        # Reset environment to the exact starting state of the expert
        continuous_state = self.reset(traj_id)

        simulated_path = []

        for step in range(max_steps):
            # Discretize the continuous state from the environment
            #continuous_state = np.delete(continuous_state, 4) # Remove front_risk from the state before discretization for ablation
            state_idx = mdp_builder.discretize_state(continuous_state)

            # Get action probabilities from the learned policy
            action_probs = policy[state_idx]

            prob_sum = np.sum(action_probs)
            if prob_sum > 0:
                action_probs = action_probs / prob_sum
            else:
                # Solo en caso de un fallo catastrófico del estado (estado jamás visto)
                action_probs = np.ones(mdp_builder.n_actions) / mdp_builder.n_actions

            # Sample an action based on the probabilities
            action_idx = np.random.choice(mdp_builder.n_actions, p=action_probs)
        
            # ¡Extraer la acción FÍSICA para el entorno DrivEnv!
            physical_action_tuple = mdp_builder.physical_actions[action_idx]
        
            # Dar el paso con los valores físicos
            next_state, reward, done, info = self.step(physical_action_tuple)

            features_dict = self._features()

            # Record the agent's physics and features
            simulated_path.append({
                "time_step": step,
                "x": info["x"],
                "y": info["y"],
                "v": features_dict["longitudinal_speed"],
                "accel": features_dict["ego_longitudinal_acc"],
                "lat_accel": features_dict["ego_lateral_acc"],
                "jerk": features_dict["ego_longitudinal_jerk"],
                "front_risk": features_dict["front_risk"],
                "collision": features_dict["collision"],
                "interaction": features_dict["interaction"]
            })
            ## Record the agent's physics
            #simulated_path.append({
            #    "time_step": step,
            #    "x": info["x"],
            #    "y": info["y"],
            #    "v": info["velocity"],
            #    "action_v": physical_action_tuple[0],
            #    "action_lane": physical_action_tuple[1]
            #})

            continuous_state = next_state
            if done:
                break

        return pd.DataFrame(simulated_path)

    def evaluate_performance(self, mdp_builder, policy, test_trajectories):
        """
        Simulates all test trajectories and compares them against the expert dataset.
        """
        #
        ### Calculo de RMSE para Velocidad, Posición X e Y
        #
        all_rmse_v = []
        all_rmse_x = []
        all_rmse_y = []
        all_fee = []  # Para almacenar los errores de feature expectation (FEE) de cada trayectoria
        all_fee_vectors = []  # store fee vectors per trajectory for final breakdown

        print(f"Evaluating {len(test_trajectories)} test trajectories")

        for traj_id, expert_traj in test_trajectories:
            horizon = len(expert_traj)

            # Simulate the agent driving
            sim_df = self.simulate_agent(mdp_builder, policy, traj_id, max_steps=horizon)

            # Extract the ground truth (expert) driving from the dataset
            expert_df = self.data[self.data["traj_id"] == traj_id].head(len(sim_df)).reset_index(drop=True)

            # Calculate RMSE (Root Mean Square Error)
            # Only compare up to the length that was actually simulated
            min_len = min(len(sim_df), len(expert_df))

            rmse_v = np.sqrt(mean_squared_error(expert_df["computed_v"].iloc[:min_len], sim_df["v"].iloc[:min_len]))
            rmse_x = np.sqrt(mean_squared_error(expert_df["x"].iloc[:min_len], sim_df["x"].iloc[:min_len]))
            rmse_y = np.sqrt(mean_squared_error(expert_df["y"].iloc[:min_len], sim_df["y"].iloc[:min_len]))

            all_rmse_v.append(rmse_v)
            all_rmse_x.append(rmse_x)
            all_rmse_y.append(rmse_y)

            # 
            # CALCULO DEL FEATURE EXPECTATION ERROR (FEE)
            # 
            # Reconstruir las características del EXPERTO a partir del dataset crudo
            th = expert_df['time_headway'].fillna(0.0).iloc[:min_len].values
            v_exp = expert_df['computed_v'].iloc[:min_len].values
            acc_exp = expert_df['longitudinal_acceleration'].iloc[:min_len].values
            lat_acc_exp = expert_df['lateral_acceleration'].iloc[:min_len].values
            jerk_exp = expert_df['longitudinal_jerk'].iloc[:min_len].values
            
            # Recrear la misma lógica física que usa el entorno
            front_risk_exp = np.where(th > 0, np.exp(-th), 0.0)
            distance_exp = np.where(th > 0, th * v_exp, 999.0)
            collision_exp = np.where(distance_exp < 2.0, 1.0, 0.0)
            interaction_exp = np.where((acc_exp < 0) & (front_risk_exp > 0.05), np.abs(acc_exp), 0.0)
            
            # Agrupar las características del experto (absolutas, igual que en MDPBuilder)
            expert_features = np.array([
                np.sum(np.abs(v_exp)),
                np.sum(np.abs(acc_exp)),
                np.sum(np.abs(lat_acc_exp)),
                np.sum(np.abs(jerk_exp)),
                np.sum(front_risk_exp),
                np.sum(collision_exp),
                np.sum(interaction_exp)
            ])
            
            # Extraer las características del AGENTE (ya calculadas en simulate_agent)
            agent_features = np.array([
                sim_df['v'].iloc[:min_len].abs().sum(),
                sim_df['accel'].iloc[:min_len].abs().sum(),
                sim_df['lat_accel'].iloc[:min_len].abs().sum(),
                sim_df['jerk'].iloc[:min_len].abs().sum(),
                sim_df['front_risk'].iloc[:min_len].sum(),
                sim_df['collision'].iloc[:min_len].sum(),
                sim_df['interaction'].iloc[:min_len].sum()
            ])
            
            # Calcular la Distancia (FEE = Norma L2 de la diferencia)
            expert_features = expert_features / min_len
            agent_features = agent_features / min_len
            fee_vector = expert_features - agent_features
            fee_scalar = np.linalg.norm(fee_vector)
            
            all_fee.append(fee_scalar)
            all_fee_vectors.append(fee_vector)
            
        avg_fee = np.mean(all_fee)
        avg_fee_vector = np.mean(np.array(all_fee_vectors), axis=0)
        feature_names = ['v', 'accel', 'lat_accel', 'jerk', 'front_risk', 'collision', 'interaction']

        print("\nEVALUATION RESULTS")
        print(f"Average Velocity RMSE: {np.mean(all_rmse_v):.2f} m/s")
        print(f"Average X-Position RMSE: {np.mean(all_rmse_x):.2f} meters")
        print(f"Average Y-Position RMSE: {np.mean(all_rmse_y):.2f} meters")
        print(f"Average FEE: {avg_fee:.2f}")
        print("Average FEE Breakdown by Feature:")
        for name, err in zip(feature_names, avg_fee_vector):
            print(f"  - {name}: {err:.4f}")

        return np.mean(all_rmse_v), np.mean(all_rmse_x), np.mean(all_rmse_y)
    

    def evaluate_style_metrics(self, mdp_builder, policy, test_trajectories):
        """
        Calculates Negative Log-Likelihood (NLL) and Wasserstein Distance 
        for driving style personalization analysis.
        """
        nll_total = 0.0
        state_count = 0
        
        all_expert_front_risk = []
        all_agent_front_risk = []
        
        for traj_id, expert_traj in test_trajectories:
            horizon = len(expert_traj)
            
            #1. WASSERSTEIN DISTANCE PREP 
            # Simulate agent to get its front_risk distribution
            sim_df = self.simulate_agent(mdp_builder, policy, traj_id, max_steps=horizon)
            
            # ¡CORRECCIÓN! Extraer el DataFrame original del experto usando el traj_id
            expert_df = self.data[self.data["traj_id"] == traj_id].reset_index(drop=True)
            min_len = min(len(sim_df), len(expert_df))
            
            # Expert front risk (recreated from physical data)
            th = expert_df['time_headway'].fillna(0.0).iloc[:min_len].values
            front_risk_exp = np.where(th > 0, np.exp(-th), 0.0)
            
            all_expert_front_risk.extend(front_risk_exp)
            all_agent_front_risk.extend(sim_df['front_risk'].iloc[:min_len].values)

            #2. NEGATIVE LOG-LIKELIHOOD (NLL) 
            # Reset env to track the expert's exact decisions
            self.reset(traj_id)
            for idx in range(1, len(self.current_traj)):
                prev = self.current_traj.iloc[idx - 1]
                curr = self.current_traj.iloc[idx]

                # What did the expert physically do
                delta_v = curr["v"] - prev["v"]
                delta_y = curr["y"] - prev["y"]
                delta_lane = int(round(delta_y / self.lane_width))

                # Discretize the state the expert was in
                state_features = self.get_state()
                if state_features is None: break
                state_idx = mdp_builder.discretize_state(state_features)

                # Map expert's physical action to the closest discrete MDP action
                dists = [ (a[0]-delta_v)**2 + (a[1]-delta_lane)**2 for a in mdp_builder.physical_actions ]
                action_idx = int(np.argmin(dists))

                # Fetch probability policy assigned to the expert's action
                prob = policy[state_idx][action_idx]
                
                # Add to NLL (using 1e-10 to prevent log(0))
                nll_total += -np.log(max(prob, 1e-10))
                state_count += 1

                # Step environment matching expert exactly to evaluate next state
                self.step([delta_v, delta_lane])

        # Calculate final metrics
        avg_nll = nll_total / max(1, state_count)
        w_distance = wasserstein_distance(all_expert_front_risk, all_agent_front_risk)
        
        return avg_nll, w_distance