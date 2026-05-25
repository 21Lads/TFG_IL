import numpy as np
import pandas as pd
import scipy.sparse as sp

class MDPBuilder:
    """
    Processes continuous M40 highway data into a discrete Markov Decision Process (MDP)
    required for Ziebart's Maximum Entropy IRL algorithm.
    """
    def __init__(self, dataset: pd.DataFrame, lane_width=2.0, dt=0.5):
        self.df = dataset
        self.dt = dt
        self.lane_width = lane_width
        
        # 1. ESPACIO DE ACCIONES: 5 de velocidad x 3 de carril = 15 acciones
        self.v_actions_physical = [-1.5, -0.75, 0.0, 0.75, 1.5] # Físico en m/s
        self.lane_actions = [-1, 0, 1]
        
        self.actions = []          # Tuplas lógicas para el IRL: (0, 1)
        self.physical_actions = [] # Valores físicos para el simulador: (-3.0, 0)
        
        for v_idx, phys_v in enumerate(self.v_actions_physical):
            for l_idx, phys_l in enumerate(self.lane_actions):
                self.actions.append((v_idx, l_idx)) 
                self.physical_actions.append((phys_v, phys_l))
                
        self.n_actions = len(self.actions)

        v_data = self.df['computed_v'].values
        
        # 2. CONTENEDORES (BINS) AJUSTADOS A METROS POR SEGUNDO
        # 7 BINS OPTIMIZADOS (Reducidos a 1512 Estados en total)
        self.bins = [
            # 1: Velocidad (7 Dims)
            np.percentile(v_data, [0, 20, 40, 60, 80, 100]),       # Data driven bins para velocidad
            
            # 2: Acel Long. (3 Dims: Frenando, Estable, Acelerando)
            np.array([-2.0, 2.0]),         
            
            # 3: Acel Lat. (3 Dims: Izquierda, Centro, Derecha)
            np.array([-0.785, 0.785]),         
            
            # 4: Jerk (3 Dims: Negativo, Suave, Positivo)
            np.array([-0.5, 0.5]),         
            
            # 5: Front Risk (2 Dims: Seguro vs Peligro)
            np.array([0.5]),        # Time headway of around 1 second(s)              
            
            # 6: Colisión (2 Dims: Falso vs Verdadero)
            np.array([0.5]),              
            
            # 7: Interacción (2 Dims: No vs Sí)
            np.array([0.5])               
        ]
        
        self.dims = [len(b) + 1 for b in self.bins] 
        self.n_states = np.prod(self.dims)
        
        self.bin_centers = []
        for b in self.bins:
            centers = (b[:-1] + b[1:]) / 2
            centers = np.insert(centers, 0, b[0] - 1.0) 
            centers = np.append(centers, b[-1] + 1.0)
            self.bin_centers.append(centers)

    def discretize_state(self, continuous_state):
        """Maps a 7D state to a single integer ID."""
        bin_indices = [np.digitize(val, self.bins[i]) for i, val in enumerate(continuous_state)]
        return int(np.ravel_multi_index(bin_indices, self.dims))

    def get_feature_matrix(self):
        """IMPLEMENTANDO LAS 7 CARACTERÍSTICAS DEL PAPER NGSIM"""
        features = np.zeros((self.n_states, 7)) 
        for s_i in range(self.n_states):
            bin_indices = np.unravel_index(s_i, self.dims)
            
            # Extraer las 7 variables del centro de los bins
            v = self.bin_centers[0][bin_indices[0]]
            accel = self.bin_centers[1][bin_indices[1]]
            lat_accel = self.bin_centers[2][bin_indices[2]]
            jerk = self.bin_centers[3][bin_indices[3]]
            front_risk = self.bin_centers[4][bin_indices[4]]
            collision = self.bin_centers[5][bin_indices[5]]
            interaction = self.bin_centers[6][bin_indices[6]]
            
            # DISEÑO NGSIM: Eficiencia, Confort y Seguridad
            speed_dev = abs(v) 
            comfort_accel = abs(accel)
            comfort_jerk = abs(jerk)
            lat_penalty = abs(lat_accel)
            
            # Guardamos las 7 métricas de coste/comportamiento
            features[s_i] = [
                speed_dev,       # 1. Velocidad (Eficiencia)
                comfort_accel,   # 2. Acel Long (Confort)
                lat_penalty,     # 3. Acel Lat (Confort)
                comfort_jerk,    # 4. Jerk (Confort)
                front_risk,      # 5. Riesgo Frontal
                collision,       # 6. Collision
                interaction      # 7. Interacción
            ]
            
        max_vals = np.max(np.abs(features), axis=0)
        max_vals[max_vals == 0] = 1 
        return features / max_vals

    def extract_trajectories(self):
        demonstrations = []
        df_sorted = self.df.sort_values(by=['traj_id', 'Time(s)'])
        
        for traj_id, group in df_sorted.groupby('traj_id'):
            group = group.reset_index(drop=True)
            traj = []
            
            for i in range(len(group) - 1):
                row, next_row = group.iloc[i], group.iloc[i + 1]
                
                # 1. Variables Físicas Base
                v = row['computed_v']
                acc = row['longitudinal_acceleration']
                lat_acc = row['lateral_acceleration']
                jerk = row['longitudinal_jerk']
                
                # 2. Calcular Riesgos (Sensores)
                # Riesgo Frontal
                th = row.get('time_headway', 0.0)
                front_risk = np.exp(-th) if th > 0 else 0.0
                
                # Colisión: Aproximar distancia (Velocidad * Time Headway)
                distance = th * v if th > 0 else 999.0
                collision = 1.0 if distance < 2.0 else 0.0 # Colisión si está a menos de 2.0m
                
                # Interacción (Si frena)
                interaction = abs(acc) if acc < 0 else 0.0
                
                # VECTOR CONTINUO 7D
                continuous_state = [v, acc, lat_acc, jerk, front_risk, collision, interaction]
                state_id = self.discretize_state(continuous_state)
                
                dv = next_row['computed_v'] - row['computed_v']
                dy = next_row['y'] - row['y']
                
                v_action_idx = int(np.argmin(np.abs(np.array(self.v_actions_physical) - dv)))
                if dy > self.lane_width / 2: l_action_idx = 2
                elif dy < -self.lane_width / 2: l_action_idx = 0
                else: l_action_idx = 1
                
                action_tuple = (v_action_idx, l_action_idx)
                action_id = self.actions.index(action_tuple)
                    
                traj.append((state_id, action_id))
                
            if traj: demonstrations.append((traj_id, traj))

        return demonstrations

    def build_transition_matrix(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        all_s = np.arange(self.n_states)
        bin_indices = np.array(np.unravel_index(all_s, self.dims))
        
        # Extraemos las 7 características
        features = np.array([self.bin_centers[f][bin_indices[f]] for f in range(7)])
        v, accel, lat_accel, jerk, front_risk, collision, interaction = features

        for a_idx, (phys_v, phys_l) in enumerate(self.physical_actions):
            new_v = np.maximum(0, v + phys_v)
            new_accel = (new_v - v) / self.dt
            new_jerk = (new_accel - accel) / self.dt
            # Lateral acceleration scaling factor based on speed
            # Assume that lateral acceleration increases with speed but has a cap
            max_lateral_accel = 0.8  # Max lateral acceleration (from your bins)
            scaling_factor = 0.05  
            # Scale lateral action with speed (clamped between -max_lateral_accel and max_lateral_accel)
            new_lat_accel = np.clip(phys_l * scaling_factor * v, -max_lateral_accel, max_lateral_accel)
            new_front_risk = np.copy(front_risk)
            new_collision = np.zeros_like(collision) # Reiniciamos la colisión a 0
            current_deceleration = np.maximum(0.0, -new_accel)
            new_interaction = np.zeros_like(interaction)
            
            #N = 4  # number of steps to complete lane change (2s with dt=0.5)
            #a_max = 0.5  # peak lateral acceleration (m/s^2)
            ## Normalize current lateral accel to detect phase of lane change
            ## (rough proxy for "progress")
            #progress = np.clip(np.abs(lat_accel) / a_max, 0.0, 1.0)
            ## Define smooth profile (triangle shape: accelerate then decelerate)
            ## early phase → ramp up, late phase → ramp down
            #ramp_up = progress < 0.5
            #target_profile = np.where(
            #    ramp_up,
            #    2 * progress,          # increasing phase
            #    2 * (1 - progress)    # decreasing phase
            #)
            ## Convert profile to actual acceleration
            #target_lat_accel = phys_l * a_max * target_profile
            ##  CRITICAL: enforce commitment 
            ## If already mid lane-change, ignore opposite action
            #in_progress = np.abs(lat_accel) > 0.05
            #flip_mask = (lat_accel * phys_l) < 0
            ## If trying to flip direction mid-change → ignore action
            #effective_action = np.where(in_progress & flip_mask, 0, phys_l)
            ## Recompute target with corrected action
            #target_lat_accel = effective_action * a_max * target_profile
            ## Smooth update
            #alpha = 0.3
            #new_lat_accel = lat_accel + alpha * (target_lat_accel - lat_accel)
            ## Snap to zero when finished
            #finished_mask = np.abs(new_lat_accel) < 0.05
            #new_lat_accel[finished_mask] = 0.0
            ## Clamp
            #new_lat_accel = np.clip(new_lat_accel, -a_max, a_max)

            mask_car_ahead = front_risk > 0.05
            
            th_current = np.zeros_like(v)
            th_current[mask_car_ahead] = -np.log(front_risk[mask_car_ahead])
            dist_current = np.full_like(v, 999.0)
            dist_current[mask_car_ahead] = th_current[mask_car_ahead] * v[mask_car_ahead]
            
            dist_new = dist_current + (v - new_v) * self.dt
            dist_new = np.maximum(0.1, dist_new) 
            
            th_new = np.zeros_like(v)
            th_new[mask_car_ahead] = dist_new[mask_car_ahead] / np.maximum(new_v[mask_car_ahead], 0.1)
            new_front_risk[mask_car_ahead] = np.exp(-th_new[mask_car_ahead])
            
            # (Balanceo de Probabilidades)
            
            if phys_v >= 0.0:
                new_collision[dist_new < 2.0] = 1.0
            # Si el coche frena (phys_v < 0), mantener new_collision = 0.0
            
            # Interaction
            new_interaction[mask_car_ahead] = current_deceleration[mask_car_ahead]
            #new_interaction = interaction
            
            next_cont = np.stack([
                new_v, new_accel, new_lat_accel, new_jerk, 
                new_front_risk, new_collision, new_interaction
            ], axis=1)
            
            next_bin_indices = []
            for f in range(7):
                indices = np.digitize(next_cont[:, f], self.bins[f])
                indices = np.clip(indices, 0, self.dims[f] - 1)
                next_bin_indices.append(indices)
            
            s_j_indices = np.ravel_multi_index(next_bin_indices, self.dims)
            P[all_s, a_idx, s_j_indices] = 1.0
            
        return P
    

    def get_initial_dist_and_horizon(self, trajectories):
        """Calculates start state probabilities and maximum trajectory length."""
        initial_dist = np.zeros(self.n_states)
        max_horizon = 0
        
        for traj_id, traj in trajectories:
            max_horizon = max(max_horizon, len(traj))
            start_state_idx = traj[0][0]
            initial_dist[start_state_idx] += 1
            
        return initial_dist / len(trajectories), max_horizon

    def build_all(self):
        """Master execution function. Returns everything needed for IRL."""
        print(f"1. Initializing MDP with {self.n_states} discrete states...")
        
        print("2. Extracting and discretizing expert trajectories...")
        trajectories = self.extract_trajectories()
        
        print("3. Building Feature Matrix phi(s)...")
        feature_matrix = self.get_feature_matrix()
        
        print("4. Calculating Initial State Distribution and Horizon...")
        initial_dist, horizon = self.get_initial_dist_and_horizon(trajectories)
        
        print("5. Building deterministic Transition Matrix P(s'|s,a)")
        transition_matrix = self.build_transition_matrix()
        
        print("MDP Build Complete.")
        
        return {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "expert_trajectories": trajectories,
            "feature_matrix": feature_matrix,
            "initial_dist": initial_dist,
            "horizon": horizon,
            "transition_matrix": transition_matrix,
            "bins": self.bins, 
            "dims": self.dims
        }