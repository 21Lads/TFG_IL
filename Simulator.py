import gymnasium as gym
import highway_env
import numpy as np
import pickle
import time
import pandas as pd 
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
import cv2

class IntegratedHighwaySimulator:
    """
    A unified environment that runs highway-env but natively 'speaks' 
    the language of trained MaxEnt IRL policy.
    """
    def __init__(self, mdp_data, render=True, dt=0.5):
        self.mdp_data = mdp_data
        self.dt = dt

        IDMVehicle.POLITENESS = 0.1  # Default is 0.3. Lower = more selfish, will cut off others to overtake.
        IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN = 0.8  # Default is 0.2. Lower = highly impatient, will overtake at the slightest slow down.
        IDMVehicle.LANE_CHANGE_DELAY = 0.5  # Faster execution of lane changes once decided.
        
        self.render_enabled = render
        self.env = gym.make("highway-v0", render_mode="rgb_array")

        self.env.unwrapped.configure({
            "offscreen_rendering": True,
            "lanes_count": 3,
            "vehicles_density": 1.8,
            "speed_limit": 33.3, # 120 km/h
            "duration": 60, # 20 seconds at 0.5s per step
            "policy_frequency": int(1 / self.dt),
            # for NPC cars
            "right_lane_reward": 0.8, # Encourages staying in the right lane when not overtaking
            "high_speed_reward": 0.8, 
            "lane_change_reward": -0.10, # Small penalty prevents erratic zig-zagging

            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [15.0, 20.0, 25.0, 30.0, 35.0], # Rango de velocidades realistas (m/s)
            },
            "ego_vehicle_trajectory": {
                "longitudinal": {
                    "acceleration_bounds": [-5.0, 5.0], # Ajustado acciones físicas
                },
                "lateral": {
                    "acceleration_bounds": [-2.0, 2.0] # Cambios de carril suaves
                }
            }
        })
        
        self.prev_v = 10.0
        self.prev_accel = 0.0
        self.prev_y_pos = 0.0
        self.prev_vy = 0.0 
        
        # Setup Action Translation Map (15 IRL Actions -> Physical Values)
        self.physical_actions = []
        for phys_v in [-1.5, -0.75, 0.0, 0.75, 1.5]:
            for phys_l in [-1, 0, 1]:
                self.physical_actions.append((phys_v, phys_l))

    def capture_frame(self):
        """Returns the current frame as a BGR image (for OpenCV)."""
        frame = self.env.render()  # already rgb_array
        
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def overlay_action_probs(self, frame, action_probs):
        
        H, W, _ = frame.shape

        lane_centers = [
            int(W * 0.3),
            int(W * 0.5),
            int(W * 0.7),
            int(W * 0.9)
        ]

        bar_bottom = int(H * 0.85)
        bar_max_height = int(H * 0.25)
        bar_width = 40

        def draw_bar(x, prob, color, label):
            h = int(prob * bar_max_height)

            top_left = (x - bar_width // 2, bar_bottom - h)
            bottom_right = (x + bar_width // 2, bar_bottom)

            cv2.rectangle(frame, top_left, bottom_right, color, -1)

            cv2.putText(frame, f"{label}:{prob:.2f}",
                        (x - 45, bar_bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Map your 15 → semantic actions (adjust if needed)
        # Show action probabilities for the speed changes [-2.0, -0.5, 0.0, 0.5, 1.2] and lane changes [-1, 0, 1]
        p_left = sum(prob for (prob, (dv, dl)) in zip(action_probs, self.physical_actions) if dl == 1)
        p_right = sum(prob for (prob, (dv, dl)) in zip(action_probs, self.physical_actions) if dl == -1)
        p_dec = sum(prob for (prob, (dv, dl)) in zip(action_probs, self.physical_actions) if dv < 0.0)
        p_acc = sum(prob for (prob, (dv, dl)) in zip(action_probs, self.physical_actions) if dv >= 0.0)
        #normalise
        total = p_left + p_right + p_dec + p_acc
        if total > 0:
            p_left /= total
            p_right /= total
            p_dec /= total
            p_acc /= total
        # now increase decimal precision of all actions
        p_left = round(p_left, 5)
        p_right = round(p_right, 5)
        p_dec = round(p_dec, 5)
        p_acc = round(p_acc, 5)

        draw_bar(lane_centers[0], p_left, (0,255,0), "LEFT")
        draw_bar(lane_centers[1], p_dec, (255,0,0), "DECC")
        draw_bar(lane_centers[2], p_right, (0,0,255), "RIGHT")
        draw_bar(lane_centers[3], p_acc, (255,255,0), "ACC")

        return frame
    
    def step_with_frame(self, irl_action_idx, action_probs=None, capture="before"):
        """
        capture:
        "before" → frame aligned with decision
        "after"  → frame after action applied
        """
    
        if capture == "before":
            frame = self.capture_frame()
            if action_probs is not None:
                frame = self.overlay_action_probs(frame, action_probs)

        # Step environment
        next_state_idx, reward, done, truncated, info = self.step(irl_action_idx)

        if capture == "after":
            frame = self.capture_frame()
            if action_probs is not None:
                frame = self.overlay_action_probs(frame, action_probs)

        return next_state_idx, reward, done, truncated, info, frame

    def reset(self):
        """Resets the environment and returns the initial discrete IRL state."""
        obs, info = self.env.reset()
        
        # Reset trackers
        self.prev_v = self.env.unwrapped.vehicle.speed
        self.prev_accel = 0.0
        
        continuous_state = self._extract_irl_state()
        return self._discretize(continuous_state)

    def step(self, irl_action_idx):
        """
        Takes an IRL action (0-14), translates it, steps the highway environment,
        and returns the next discrete IRL state.
        """
        # 1. Translate Action
        highway_action = self._translate_action(irl_action_idx)
        
        # 2. Step Environment
        obs, reward, done, truncated, info = self.env.step(highway_action)
        
        # 3. Extract and Discretize New State
        continuous_state = self._extract_irl_state()
        next_state_idx = self._discretize(continuous_state)
        
        return next_state_idx, reward, done, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    #  INTERNAL TRANSLATION METHODS 

    def _extract_irl_state(self):
        ego_vehicle = self.env.unwrapped.vehicle
        ego_speed = ego_vehicle.speed
        position = ego_vehicle.position
        current_vy = 0.0
        lat_accel = 0.0
        accel = (ego_speed - self.prev_v) / self.dt
        jerk = (accel - self.prev_accel) / self.dt
        if self.prev_y_pos != 0.0:
            # Velocidad lateral (metros / segundo)
            current_vy = (position[1] - self.prev_y_pos) / self.dt
            # Aceleración lateral (m / s^2)
            lat_accel = (current_vy - self.prev_vy) / self.dt
        
        self.prev_v = ego_speed
        self.prev_accel = accel
        self.prev_y_pos = position[1]
        self.prev_vy = current_vy
        
        front_risk = 0.0
        collision = 0.0
        interaction = 0.0
        
        front_vehicle, rear_vehicle = self.env.unwrapped.road.neighbour_vehicles(ego_vehicle, lane_index=ego_vehicle.lane_index)
        
        if front_vehicle:
            distance_front = front_vehicle.position[0] - ego_vehicle.position[0]
            if distance_front < 2.0: 
                collision = 1.0 
                
            if ego_speed > 0.1:
                th_front = distance_front / ego_speed
                if th_front > 0:
                    front_risk = np.exp(-th_front)
            
            if front_vehicle.speed < ego_speed:
                interaction = abs(accel) if accel < 0 else 0.0
                #interaction = abs(ego_speed - front_vehicle.speed) / self.dt
                
        #if rear_vehicle:
        #    distance_rear = ego_vehicle.position[0] - rear_vehicle.position[0]
        #    # Riesgo trasero se basa en la velocidad del coche que viene detrás
        #    if rear_vehicle.speed > 0.1:
        #        th_rear = distance_rear / rear_vehicle.speed
        #        if th_rear > 0:
        #            rear_risk = np.exp(-th_rear)

        return [ego_speed, accel, lat_accel, jerk, front_risk, collision, interaction]

    def _translate_action(self, irl_action_idx):
        """Maps 15-action IRL space to 5-action Highway space since Highway-env only has 5 discrete actions."""
        delta_v, delta_lane = self.physical_actions[irl_action_idx]
        
        if delta_lane == 1: return 0 # Lane Left
        if delta_lane == -1:  return 2 # Lane Right
        if delta_v > 0.7:    return 3 # Faster
        if delta_v < -0.7:   return 4 # Slower
        return 1 # Idle

    def _discretize(self, continuous_state):
        """Matches the continuous features to the MDP's discrete state ID."""
        bins = self.mdp_data["bins"]
        dims = self.mdp_data["dims"]
        
        bin_indices = []
        for i in range(len(continuous_state)):
            val = continuous_state[i]
            idx = np.digitize(val, bins[i])
            # Cap to prevent out-of-bounds in the new simulator
            bin_indices.append(min(idx, dims[i] - 1))
            
        return np.ravel_multi_index(bin_indices, dims)



if __name__ == "__main__":

    #with open("results/styles_artifacts.pkl", "rb") as f:
    #    styles = pickle.load(f)
    #    
    ## Extract the separate profiles
    #agg_mdp_data = styles["aggressive"]["mdp_data"]
    #agg_policy = styles["aggressive"]["policy"]
    #
    #cons_mdp_data = styles["conservative"]["mdp_data"]
    #cons_policy = styles["conservative"]["policy"]

    #sim = IntegratedHighwaySimulator(mdp_data=cons_data, render=True)

    print("Loading IRL Artifacts and Expert Data...")
    try:
        with open("results/irl_artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)
            
        mdp_data = artifacts["mdp_data"]
        policy = artifacts["policy"]
        expert_df = pd.read_csv("results/expert_trajectory.csv")
        
    except FileNotFoundError:
        print(" Error: Missing files in 'results/' folder.")
        #print file names that are expected
        print(" Expected: 'irl_artifacts.pkl' and 'expert_trajectory.csv'")
        exit()

    print("Initializing Unified Simulator...")
    sim = IntegratedHighwaySimulator(mdp_data=mdp_data, render=True)
    
    state_idx = sim.reset()
    done = truncated = False
    
    expert_start_x = expert_df.iloc[0]['x']
    expert_start_y = expert_df.iloc[0]['y']
    expert_start_v = expert_df.iloc[0]['computed_v']
    
    # M40: 500, 503, 506 (Ancho de 3m) -> Índices: 0, 1, 2 -> Highway-Env: 0, 4, 8 (Ancho de 4m)
    def normalize_y(y_real):
        lane_idx = round((y_real - 500.0) / 3.0)
        lane_idx = max(0, min(2, lane_idx)) 
        return lane_idx * 4.0

    sim_start_y = normalize_y(expert_start_y)
    
    sim.env.unwrapped.vehicle.position = np.array([0.0, sim_start_y])
    sim.env.unwrapped.vehicle.speed = expert_start_v
    
    ghost_car = Vehicle(
        road=sim.env.unwrapped.road, 
        position=[10.0, sim_start_y], 
        speed=expert_start_v
    )
    ghost_car.color = (255, 0, 255) # Magenta
    ghost_car.crashed = False 
    #sim.env.unwrapped.road.vehicles.append(ghost_car)
    
    print("\n--- Starting Ghost Car Comparison (Shadow Mode) ---")
    step_count = 0
    paused = True
    # IGNORE CRASHES with "while not truncated:"
    while not truncated and not done:

        frame = sim.capture_frame()

    # Show frame
        cv2.imshow("Simulator", frame)

        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key == ord('q'):
            break

        # SPACE = pause/unpause
        if key == ord(' '):
            action_probs = policy[state_idx]
            paused = not paused

        # 'n' = step ONE frame forward 
        if key == ord('n'):
            action_probs = policy[state_idx]

        if key == ord('p'):  # "p" for probe
            action_probs = policy[state_idx]

            frame_with_overlay = sim.overlay_action_probs(frame.copy(), action_probs)

            cv2.imshow("Probe", frame_with_overlay)
            cv2.waitKey(0)

        # overlay BEFORE stepping (decision frame)
        frame = sim.overlay_action_probs(frame, action_probs)
        cv2.imshow("Simulator", frame)
        cv2.waitKey(1)

        action_idx = np.random.choice(len(action_probs), p=action_probs)
        next_state_idx, reward, done, truncated, info = sim.step(action_idx)
        # Ask IRL Agent for Action
        action_probs = policy[state_idx]

        prob_sum = np.sum(action_probs)
        #if prob_sum > 0:
        #    action_probs = action_probs / prob_sum
        #else:
        #    # PROFESSIONAL FALLBACK: Maintain current trajectory rather than panicking arbitrarily
        #    action_probs = np.zeros(len(action_probs))
        #    idle_action_idx = sim.physical_actions.index((0.0, 0)) # Find the (0 delta_v, 0 delta_lane) action
        #    action_probs[idle_action_idx] = 1.0

        #action_idx = np.random.choice(len(action_probs), p=action_probs)
        #print(f"Step {step_count}: State {state_idx} -> Action {action_idx}: {sim.physical_actions[action_idx]} (Action Probabilities: {action_probs})")
        
        if not paused:
            action_probs = policy[state_idx]
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            state_idx, reward, done, truncated, info = sim.step(action_idx)

        sim.env.unwrapped.vehicle.crashed = False
        #ghost_car.crashed = False
        
        if step_count < len(expert_df):
            historical_data = expert_df.iloc[step_count]
            
            # Avanzar X relativo al punto de partida original
            normalized_x = historical_data['x'] - expert_start_x
            # Calcular Y con nuestra nueva función de carriles
            normalized_y = normalize_y(historical_data['y'])
            
            ghost_car.position = np.array([normalized_x, normalized_y])
            ghost_car.speed = historical_data['computed_v']
            
        #sim.render()
        time.sleep(0.1) 
        
        state_idx = next_state_idx
        step_count += 1
            
    sim.close()
    print("Simulation Complete")


    