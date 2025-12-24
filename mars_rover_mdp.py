"""Mars Rover MDP - Value Iteration Algorithm

Mini-Project: The Mars Rover MDP
Course: Artificial Intelligence
Topic: Markov Decision Processes (MDP) & Dynamic Programming

Project Overview:
In this project, we design an autonomous agent for a Mars Rover. The Rover must balance
the need to collect scientific data (Drill) and transmit it (Transmit) against the risk
of running out of battery (Harvest energy).

The Challenge: The environment is stochastic. Attempting to charge the battery (Harvest)
might fail due to dust storms, and drilling consumes significant energy. The goal is to
implement Value Iteration to find the optimal policy π*.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random

# Configuration for plots
plt.style.use('seaborn-v0_8-whitegrid')


class MarsRoverEnv:
    """Mars Rover Environment for MDP learning.
    
    State Space: Battery levels 0 to 100
    Action Space: Harvest, Drill, Transmit
    """
    
    def __init__(self, storm_prob=0.2, gamma=0.9):
        """Initialize the Mars Rover Environment.
        
        Args:
            storm_prob: Probability that Harvest action fails due to dust storm
            gamma: Discount factor
        """
        self.states = np.arange(0, 101, 10)  # 0, 10, 20 ... 100
        self.actions = ['Harvest', 'Drill', 'Transmit']
        self.storm_prob = storm_prob  # Probability harvest fails
        self.gamma = gamma  # Discount factor
        
        # Costs and Rewards
        self.R_drill = 10
        self.R_transmit = 5
        self.R_harvest = 0
        self.R_death = 0  # Penalty for reaching 0 battery
    
    def get_transition_reward(self, state, action):
        """Returns a list of tuples: (probability, next_state, reward)
        
        Args:
            state: Current battery level
            action: Action to take (Harvest, Drill, or Transmit)
            
        Returns:
            List of (probability, next_state, reward) tuples
        """
        # Terminal State: If battery is 0, game over, no actions possible
        if state == 0:
            return [(1.0, 0, 0)]
        
        transitions = []
        
        if action == 'Harvest':
            # Outcome 1: Sun is clear (charge battery)
            # Logic: Increase battery by 20 (capped at 100)
            next_s_success = min(100, state + 20)
            transitions.append((1 - self.storm_prob, next_s_success, self.R_harvest))
            
            # Outcome 2: Dust storm (no charge)
            # Logic: Battery stays same
            transitions.append((self.storm_prob, state, self.R_harvest))
        
        elif action == 'Drill':
            # Logic: Costs 30 battery.
            # If state >= 30, do it. Else, penalty -1 and state stays same.
            if state >= 30:
                next_state = state - 30
                transitions.append((1.0, next_state, self.R_drill))
            else:
                transitions.append((1.0, state, -1))  # Penalty for insufficient battery
        
        elif action == 'Transmit':
            # Logic: Costs 10 battery.
            # If state >= 10, do it. Else, penalty -1 and state stays same.
            if state >= 10:
                next_state = state - 10
                transitions.append((1.0, next_state, self.R_transmit))
            else:
                transitions.append((1.0, state, -1))  # Penalty for insufficient battery
        
        return transitions


def run_simulation(env, policy_map, start_state=100, steps=20):
    """Run a simulation of the Mars Rover following a given policy.
    
    Args:
        env: The MarsRoverEnv environment
        policy_map: Dictionary mapping states to actions
        start_state: Initial battery level
        steps: Maximum number of steps to simulate
        
    Returns:
        DataFrame with simulation history and total reward
    """
    state = start_state
    history = {'step': [], 'state': [], 'action': [], 'reward': []}
    total_reward = 0
    
    for t in range(steps):
        action = policy_map[state]
        transitions = env.get_transition_reward(state, action)
        if not transitions:
            break  # Error check
        
        probs = [t[0] for t in transitions]
        candidates = [i for i in range(len(transitions))]
        chosen_idx = np.random.choice(candidates, p=probs)
        _, next_state, reward = transitions[chosen_idx]
        
        history['step'].append(t)
        history['state'].append(state)
        history['action'].append(action)
        history['reward'].append(reward)
        total_reward += reward
        state = next_state
        if state == 0:
            break
    
    return pd.DataFrame(history), total_reward


def value_iteration(env, theta=0.0001):
    """Value Iteration algorithm to find the optimal policy.
    
    Args:
        env: The MarsRoverEnv environment
        theta: Convergence threshold
        
    Returns:
        Dictionary mapping states to optimal values
    """
    # Initialize Value Function
    V = {s: 0 for s in env.states}
    
    iteration = 0
    deltas = []
    
    while True:
        delta = 0
        new_V = V.copy()
        
        for s in env.states:
            if s == 0:
                continue
            
            action_values = []
            
            # Iterate over all possible actions
            for a in env.actions:
                transitions = env.get_transition_reward(s, a)
                
                # Calculate Q(s,a) using Bellman Expectation
                # Q(s,a) = sum( P(s'|s,a) * [R + gamma * V(s')] )
                q_value = 0
                for prob, next_state, reward in transitions:
                    q_value += prob * (reward + env.gamma * V[next_state])
                
                action_values.append(q_value)
            
            # Update Value Function with the best action value
            best_value = max(action_values)
            new_V[s] = best_value
            delta = max(delta, abs(V[s] - best_value))
        
        V = new_V
        iteration += 1
        deltas.append(delta)
        
        if delta < theta:
            print(f"✓ Converged in {iteration} iterations.")
            break
    
    return V, deltas


def extract_policy(env, V):
    """Extract the optimal policy from the value function.
    
    Args:
        env: The MarsRoverEnv environment
        V: Dictionary mapping states to their optimal values
        
    Returns:
        Dictionary mapping states to best actions
    """
    policy = {}
    
    for s in env.states:
        if s == 0:
            policy[s] = None
            continue
        
        best_action = None
        best_q = -float('inf')
        
        for a in env.actions:
            transitions = env.get_transition_reward(s, a)
            q_value = 0
            for prob, next_state, reward in transitions:
                q_value += prob * (reward + env.gamma * V[next_state])
            
            if q_value > best_q:
                best_q = q_value
                best_action = a
        
        policy[s] = best_action
    
    return policy


def print_value_function(V, env):
    """Print the value function in a nice format.
    
    Args:
        V: Dictionary mapping states to values
        env: The MarsRoverEnv environment
    """
    print("\n" + "="*50)
    print("Optimal Value Function V(s)")
    print("="*50)
    for state in sorted(env.states):
        if state == 0:
            print(f"V({state:3d}) = TERMINAL")
        else:
            print(f"V({state:3d}) = {V[state]:8.4f}")
    print("="*50)


def print_policy(policy, env):
    """Print the optimal policy in a nice format.
    
    Args:
        policy: Dictionary mapping states to actions
        env: The MarsRoverEnv environment
    """
    print("\n" + "="*50)
    print("Optimal Policy π*(s)")
    print("="*50)
    for state in sorted(env.states):
        if state == 0:
            print(f"π*({state:3d}) = END")
        else:
            print(f"π*({state:3d}) = {policy[state]:10s}")
    print("="*50)


if __name__ == "__main__":
    print("\n" + "#"*50)
    print("# Mars Rover MDP - Value Iteration")
    print("#"*50)
    
    # Initialize the environment
    env = MarsRoverEnv(storm_prob=0.2, gamma=0.9)
    print(f"\n✓ Environment initialized")
    print(f"  States: {list(env.states)}")
    print(f"  Actions: {env.actions}")
    print(f"  Storm Probability: {env.storm_prob}")
    print(f"  Discount Factor: {env.gamma}")
    
    # Run Value Iteration
    print(f"\n→ Running Value Iteration...")
    V, deltas = value_iteration(env, theta=0.0001)
    
    # Extract optimal policy
    print(f"\n→ Extracting optimal policy...")
    policy = extract_policy(env, V)
    
    # Print results
    print_value_function(V, env)
    print_policy(policy, env)
    
    # Run simulations
    print(f"\n→ Running simulations with optimal policy...")
    total_rewards = []
    successes = 0
    
    for i in range(10):
        df, total_reward = run_simulation(env, policy, start_state=100, steps=50)
        total_rewards.append(total_reward)
        if total_reward > 0:
            successes += 1
    
    print(f"\n✓ Simulation Results (10 episodes):")
    print(f"  Average Reward: {np.mean(total_rewards):.2f}")
    print(f"  Successful Episodes: {successes}/10")
    print(f"  Max Reward: {np.max(total_rewards):.2f}")
    print(f"  Min Reward: {np.min(total_rewards):.2f}")
