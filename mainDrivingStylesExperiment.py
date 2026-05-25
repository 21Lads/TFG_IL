import os
import random
import numpy as np
import pandas as pd
import build_trajectories2 as build
import DrivEnv as env
import maxEntIRL as me
import MDPBuilder as mdp
import pickle

class DrivingStyleExperiment:
    """
    Clase dedicada a ejecutar el Experimento 2 de la Tesis:
    Extracción y validación cruzada de perfiles Agresivos vs Conservadores.
    """

    def __init__(self, preprocessed_data_path, trajectories_output_path):
        self.preprocessed_path = preprocessed_data_path
        self.traj_output_path = trajectories_output_path
        
        # Inicializar el constructor de trayectorias
        self.builder = build.BuildTrajectories(self.preprocessed_path, self.traj_output_path)

    def train_policy_for_style(self, style_name, vehicle_ids):
        """
        Método auxiliar que construye el MDP, divide en train/test y entrena 
        la política para un grupo específico de vehículos.
        """
        print(f"\n[{style_name.upper()}]")
        
        # Construir dataframe solo para estos vehículos
        df_style = self.builder.build_trajectories(vehicle_ids)
        
        # Crear entorno y MDP
        env_instance = env.DrivingEnv(df_style)
        mdp_builder = mdp.MDPBuilder(df_style)
        mdp_data = mdp_builder.build_all()
        
        # Split Train/Test (80/20)
        all_trajectories = mdp_data["expert_trajectories"]
        random.seed(42)
        random.shuffle(all_trajectories)
        split_idx = int(0.8 * len(all_trajectories))
        
        train_trajectories = all_trajectories[:split_idx]
        test_trajectories = all_trajectories[split_idx:]
        
        # Extraer distribuciones
        initial_dist, train_horizon = mdp_builder.get_initial_dist_and_horizon(train_trajectories)
        just_train_trajs = [traj for traj_id, traj in train_trajectories]
        
        print(f"[{style_name.upper()}] Entrenando MaxEnt IRL (Train: {len(train_trajectories)}, Test: {len(test_trajectories)})...")
        irl_agent = me.MaxEntIRL(
            n_states=mdp_data["n_states"],
            n_actions=mdp_data["n_actions"],
            transition_probs=mdp_data["transition_matrix"],
            feature_matrix=mdp_data["feature_matrix"],
            initial_state_dist=initial_dist,
            horizon=train_horizon,
            learning_rate=0.01
        )
        
        # Entrenar
        learned_theta, learned_policy = irl_agent.train(expert_trajectories=just_train_trajs, iterations=60)
        print(f"[{style_name.upper()}] Pesos Aprendidos: {learned_theta}")
        
        return env_instance, mdp_builder, learned_policy, test_trajectories, mdp_data, learned_theta

    def run_cross_validation(self):
        """
        Ejecuta el pipeline completo: Extrae los grupos, entrena los modelos y 
        realiza la evaluación cruzada (NLL y Wasserstein).
        """
        print("Driving Style Experiment: Cross-Validation of Aggressive vs Conservative Profiles")
        
        # 1. Extraer IDs (Requiere que select_driving_styles esté en build_trajectories2.py)
        agg_ids, cons_ids = self.builder.select_driving_styles()
        
        # 2. Entrenar Política Agresiva
        env_agg, mdp_agg, policy_agg, test_agg, mdp_data_agg, theta_agg = self.train_policy_for_style("Aggresive", agg_ids)
        
        # 3. Entrenar Política Conservadora
        env_cons, mdp_cons, policy_cons, test_cons, mdp_data_cons, theta_cons = self.train_policy_for_style("Conservative", cons_ids)
        
        # 4. Evaluación Cruzada
        print("\n")
        print("   RESULTS OF CROSS-VALIDATION (NLL & Wasserstein)  ")
        print("")

        # A. Agresivo con Agresivo
        print("\nEvaluando: AGGRESIVE Data with AGGRESIVE policy")
        nll_aa, w_aa = env_agg.evaluate_style_metrics(mdp_agg, policy_agg, test_agg)
        print(f" -> NLL: {nll_aa:.4f} | Distancia Wasserstein TTC: {w_aa:.4f}")

        # B. Agresivo con Conservador
        print("\nEvaluando: AGGRESIVE Data with CONSERVATIVE policy")
        # Usamos env_agg (los datos agresivos), pero mdp_cons y policy_cons
        nll_ac, w_ac = env_agg.evaluate_style_metrics(mdp_cons, policy_cons, test_agg)
        print(f" -> NLL: {nll_ac:.4f} | Distancia Wasserstein TTC: {w_ac:.4f}")

        # C. Conservador con Conservador
        print("\nEvaluando: CONSERVATIVE Data with CONSERVATIVE policy")
        nll_cc, w_cc = env_cons.evaluate_style_metrics(mdp_cons, policy_cons, test_cons)
        print(f" -> NLL: {nll_cc:.4f} | Distancia Wasserstein TTC: {w_cc:.4f}")

        # D. Conservador con Agresivo
        print("\nEvaluando: CONSERVATIVE Data with AGGRESIVE policy")
        nll_ca, w_ca = env_cons.evaluate_style_metrics(mdp_agg, policy_agg, test_cons)
        print(f" -> NLL: {nll_ca:.4f} | Distancia Wasserstein TTC: {w_ca:.4f}")

        self.save_artifacts(env_agg, mdp_agg, mdp_data_agg, theta_agg, policy_agg, test_agg,
                            env_cons, mdp_cons, mdp_data_cons, theta_cons, policy_cons, test_cons)
        
        print("\nExperiment Finalized.")

    
    def save_artifacts(self, env_agg, builder_agg, mdp_agg, theta_agg, policy_agg, test_agg,
                             env_cons, builder_cons, mdp_cons, theta_cons, policy_cons, test_cons):
        """
        Subprograma dedicado a simular una trayectoria de ejemplo para cada perfil y
        guardar todos los artefactos (DataFrames y Pickle) para su uso en Simulator y Visualiser.
        """
        print("\n")
        print("   SAVING DRIVING STYLE ARTIFACTS FOR VISUALIZATION ")
        print("\n")
        
        os.makedirs("results", exist_ok=True)

        #  Aggressive Sample 
        sample_id_agg, sample_traj_agg = test_agg[0]
        horizon_agg = len(sample_traj_agg)
        exp_df_agg = env_agg.data[env_agg.data["traj_id"] == sample_id_agg].head(horizon_agg).reset_index(drop=True)
        sim_df_agg = env_agg.simulate_agent(builder_agg, policy_agg, sample_id_agg, max_steps=horizon_agg)
        
        exp_df_agg.to_csv("results/aggressive_expert_trajectory.csv", index=False)
        sim_df_agg.to_csv("results/aggressive_simulated_trajectory.csv", index=False)

        #  Conservative Sample 
        sample_id_cons, sample_traj_cons = test_cons[0]
        horizon_cons = len(sample_traj_cons)
        exp_df_cons = env_cons.data[env_cons.data["traj_id"] == sample_id_cons].head(horizon_cons).reset_index(drop=True)
        sim_df_cons = env_cons.simulate_agent(builder_cons, policy_cons, sample_id_cons, max_steps=horizon_cons)
        
        exp_df_cons.to_csv("results/conservative_expert_trajectory.csv", index=False)
        sim_df_cons.to_csv("results/conservative_simulated_trajectory.csv", index=False)

        #  Save a dual-style pickle object 
        styles_artifacts = {
            "aggressive": {
                "theta": theta_agg,
                "mdp_data": mdp_agg,
                "test_trajectories": test_agg,
                "policy": policy_agg
            },
            "conservative": {
                "theta": theta_cons,
                "mdp_data": mdp_cons,
                "test_trajectories": test_cons,
                "policy": policy_cons
            }
        }

        with open("results/styles_artifacts.pkl", "wb") as f:
            pickle.dump(styles_artifacts, f)
            
        print("Driving styles saved successfully to 'results/styles_artifacts.pkl'!")


if __name__ == "__main__":
    # Asegúrate de poner tus rutas correctas aquí
    PREPROCESSED_PATH = r"data\dataset.csv"
    OUTPUT_PATH = r"data\M40_d07_h08_styles_trajectories.csv"
    
    experiment = DrivingStyleExperiment(PREPROCESSED_PATH, OUTPUT_PATH)
    experiment.run_cross_validation()