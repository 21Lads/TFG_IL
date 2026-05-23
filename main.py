# Run preproces, then build, then create the environment and simulate
import preprocess as pp
import build_trajectories2 as build
import DrivEnv as env
import os
import pandas as pd
import numpy as np
import maxEntIRL as me
import MDPBuilder as mdp

if __name__ == "__main__":
    # Step 1: Preprocess raw data
    raw_data_path = r"data\M40-d07-h08.dat"
    preprocessed_output_path = r"data\dataset.csv"
    #pp.Preprocess(raw_data_path, preprocessed_output_path)

    # Step 2: Build trajectories from preprocessed data
    trajectories_output_path = r"data\M40_d07_h08_trajectories_updated.csv"
    builder = build.BuildTrajectories(preprocessed_output_path, trajectories_output_path)
    trajectories_df = builder.run(trajectories_output_path)
    #trajectories_df = pd.read_csv(trajectories_output_path)

    # Step 3: Create environment and simulate
    env_instance = env.DrivingEnv(trajectories_df)
    mdp_builder = mdp.MDPBuilder(trajectories_df)
    mdp_data = mdp_builder.build_all()

    # 4. Train/Test Split (80% Train, 20% Test)
    all_trajectories = mdp_data["expert_trajectories"]
    
    # Shuffle the trajectories to ensure a random distribution
    import random
    random.seed(42)
    random.shuffle(all_trajectories)
    
    split_idx = int(0.8 * len(all_trajectories))
    train_trajectories = all_trajectories[:split_idx]
    test_trajectories = all_trajectories[split_idx:]
    
    print(f"Data Split: {len(train_trajectories)} Train | {len(test_trajectories)} Test")
    
    # Recalculate initial distribution & horizon specifically for the training set
    initial_dist, train_horizon = mdp_builder.get_initial_dist_and_horizon(train_trajectories)
    
    # 5. Initialize and Train the MaxEnt IRL Agent
    print("Initializing MaxEnt IRL...")
    irl_agent = me.MaxEntIRL(
        n_states=mdp_data["n_states"],
        n_actions=mdp_data["n_actions"],
        transition_probs=mdp_data["transition_matrix"],
        feature_matrix=mdp_data["feature_matrix"],
        initial_state_dist=initial_dist,
        horizon=train_horizon,
        learning_rate=0.01
    )
    
    print("Starting MaxEnt IRL Training Loop...")
    # Strip the traj_id off the train set before passing to IRL
    just_train_trajs = [traj for traj_id, traj in train_trajectories]
    
    learned_theta, learned_policy = irl_agent.train(
        expert_trajectories=just_train_trajs, 
        iterations=60
    )
    print("Training Completed. Learned Reward Weights:", learned_theta)
    
    # 6. Evaluate the Learned Policy in the Environment
    print("\nStarting Simulation & Evaluation on Test Set...")
    
    env_instance.evaluate_performance(
        mdp_builder=mdp_builder,
        policy=learned_policy,
        test_trajectories=test_trajectories
    )

    # 7. Save Artifacts for Visualization
    
    print(" SAVE ARTIFACTS FOR VISUALIZATION ")
    
    import pickle
    import os

    # Create a directory to store the results
    os.makedirs("results", exist_ok=True)

    # 8. Prepare the sample trajectories for the visualizer
    sample_traj_id, sample_expert_traj = test_trajectories[0]
    horizon = len(sample_expert_traj)
    sample_expert_df = env_instance.data[env_instance.data["traj_id"] == sample_traj_id].head(horizon).reset_index(drop=True)
    sample_sim_df = env_instance.simulate_agent(mdp_builder, learned_policy, sample_traj_id, max_steps=horizon)

    # 9. Save the DataFrames as CSV files (easy to read and debug)
    sample_expert_df.to_csv("results/expert_trajectory.csv", index=False)
    sample_sim_df.to_csv("results/simulated_trajectory.csv", index=False)

    # 10. Save the Python objects (Theta, MDP Data, and Test Trajectories) using Pickle
    artifacts = {
        "theta": learned_theta,
        "policy": learned_policy,
        "mdp_data": mdp_data,
        "test_trajectories": test_trajectories
    }
    
    with open("results/irl_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        

    print(" All results saved successfully in the 'results/' folder")