from sklearn.covariance import log_likelihood

import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import random

    # extract state action pairs (expert trajectories) from the trajectories M40_d07_h08_trajectories_updated.csv with features 
    # being x, y, v, time_headway, longitudinal_acceleration, lateral_acceleration and longitudinal_jerk
    # the action is lane change and velocity change
    # the state is the current position, velocity and acceleration of the ego vehicle, and the time headway to the car ahead

class MaxEntIRL:
    def __init__(self, n_states, n_actions, transition_probs, feature_matrix, initial_state_dist, horizon, learning_rate=0.01):
        """
        Args:
            n_states (int): Total discrete states
            n_actions (int): Total discrete actions
            transition_probs (np.ndarray): Shape (n_states, n_actions, n_states) -> P^a_{i,j}
            feature_matrix (np.ndarray): Shape (n_states, n_features) -> phi(s_i)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = transition_probs
        self.features = feature_matrix
        self.n_features = feature_matrix.shape[1]
        self.initial_state_dist = initial_state_dist
        self.horizon = horizon
        
        self.lr = learning_rate
        self.theta = np.random.uniform(-1, 1, self.n_features) # Weights

    def compute_expert_feature_expectations(self, expert_trajectories):
        """Calculate the empirical feature counts from the dataset."""
        expert_features = np.zeros(self.n_features)
        for traj in expert_trajectories:
            for state, _ in traj:
                expert_features += self.features[state]
        return expert_features / len(expert_trajectories)

    def backward_pass(self, reward, horizon):
        """
        Backward Pass (Vectorized)
        """
        Z_s = np.ones(self.n_states)
        
        # Precompute the exponentiated rewards
        # shape: (n_states, 1) to allow broadcasting across actions
        exp_reward = np.exp(reward)[:, np.newaxis] 
        
        for _ in range(horizon):
            # 1. Expected Z for the next state
            # np.einsum multiplies P(i, a, j) with Z_s(j) and sums over j
            # 'iaj, j -> ia' means: (states, actions, next_states) * (next_states) -> (states, actions)
            expected_Z = np.einsum('iaj,j->ia', self.P, Z_s)
            
            # 2. Compute Z_a and Z_s
            Z_a = exp_reward * expected_Z
            Z_s = np.sum(Z_a, axis=1)
            
        # 3. Compute policy with a safety check to avoid division by zero
        Z_s_safe = np.where(Z_s == 0, 1e-10, Z_s) # Prevent NaN errors
        policy = Z_a / Z_s_safe[:, np.newaxis]
        
        return policy

    def forward_pass(self, policy, initial_state_dist, horizon):
        """
        Forward Pass (Vectorized)
        """
        D_sit = np.zeros((self.n_states, horizon))
        D_sit[:, 0] = initial_state_dist
        
        for t in range(1, horizon):
            # 1. Compute joint probability of being in state 'j' and taking action 'a'
            # shape: (n_states, n_actions)
            state_action_prob = D_sit[:, t-1][:, np.newaxis] * policy
            
            # 2. Propagate to the next state 'i'
            # 'ja, jai -> i' means: (prev_states, actions) * (prev_states, actions, current_states) 
            # It sums over 'j' (prev_states) and 'a' (actions) to output 'i' (current_states)
            D_sit[:, t] = np.einsum('ja,jai->i', state_action_prob, self.P)
                        
        # Total Expected State Visitation Frequencies
        D_s = np.sum(D_sit, axis=1)
        return D_s

    def train(self, expert_trajectories, iterations=100):
        """
        Executes the full Maximum Entropy IRL loop.
        """
        expert_feat_exp = self.compute_expert_feature_expectations(expert_trajectories)
        grad_history = []
        
        for iteration in range(iterations):
          
            # Compute current reward for all states: R = theta^T * phi
            reward = np.dot(self.features, self.theta)
            
            # Run Algorithm 1 Passes, N iterations?
            policy = self.backward_pass(reward, self.horizon)
            
            D_s = self.forward_pass(policy, self.initial_state_dist, self.horizon)
            
            # Compute expected feature counts: sum_i ( D_{s,i} * phi_{s,i} )
            agent_feat_exp = D_s.dot(self.features) 
            
            # Gradient Ascent Update
            gradient = expert_feat_exp - agent_feat_exp
            self.theta += self.lr * gradient
            self.theta[-2] = -10.0 # Fixing the weight for collision to a large negative value to encourage safety
            gradient_norm = np.linalg.norm(gradient)
            if iteration % 10 == 0 or iteration == iterations - 1:
                print(f"Iteration {iteration} | Gradient Norm: {gradient_norm:.4f}")
            grad_history.append(gradient_norm)

        plt.plot(grad_history)
        plt.xlabel("Iteration")
        plt.ylabel("Gradient Norm")
        plt.title("Training Progress")
        plt.show()

        #now show log graph
        plt.plot(grad_history)
        plt.yscale('log')
        plt.xlabel("Iteration")
        plt.ylabel("Gradient Norm (log scale)")
        plt.title("Training Progress (Log Scale)")
        plt.show()

        return self.theta, policy


