# Mars Rover MDP: Value Iteration Algorithm

![Mars Rover](https://img.shields.io/badge/AI%20Course-MDP-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This project implements a **Mars Rover autonomous agent** using **Markov Decision Processes (MDP)** and **Dynamic Programming**. The rover must balance three competing objectives:

- **Harvest**: Charge the battery (+20 energy, but may fail due to dust storms)
- **Drill**: Collect scientific data (-30 energy, +10 reward)
- **Transmit**: Send data back to Earth (-10 energy, +5 reward)

The challenge is to find an **optimal policy** that maximizes long-term expected rewards while managing a stochastic environment.

## Project Structure

```
.
├── mars_rover_mdp.py       # Main implementation
├── README.md               # This file
└── requirements.txt        # Dependencies
```

## Key Concepts

### 1. **Bellman Equations**

**Bellman Expectation Equation** (value under a fixed policy π):
$$V_{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_{\pi}(s') \right]$$

**Bellman Optimality Equation** (optimal value):
$$V^*(s) = \max_{a} \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

### 2. **Value Iteration Algorithm**

Value Iteration is a **dynamic programming** algorithm that iteratively improves the value function until convergence:

```
while max(|V_new - V_old|) > θ:
    for each state s:
        V(s) = max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + γV(s')]
```

### 3. **Optimal Policy Extraction**

Once the value function converges, extract the greedy policy:

$$\pi^*(s) = \arg\max_{a} \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V(s') \right]$$

## Installation

```bash
git clone https://github.com/VictimPickle/MDP-Mars-Rover.git
cd MDP-Mars-Rover
pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline

```bash
python mars_rover_mdp.py
```

This will:
1. Initialize the Mars Rover environment
2. Run Value Iteration to compute the optimal value function
3. Extract the optimal policy
4. Display results and run simulations

### Use in Your Code

```python
from mars_rover_mdp import MarsRoverEnv, value_iteration, extract_policy, run_simulation

# Create environment
env = MarsRoverEnv(storm_prob=0.2, gamma=0.9)

# Find optimal policy
V, deltas = value_iteration(env, theta=0.0001)
optimal_policy = extract_policy(env, V)

# Run simulation
history_df, total_reward = run_simulation(env, optimal_policy, start_state=100, steps=50)
print(f"Total Reward: {total_reward}")
```

## Results

The algorithm learns the following optimal policy:

| Battery Level | Optimal Action | Reason |
|---------------|----------------|--------|
| 0-9           | END            | Terminal (dead rover) |
| 10-29         | Transmit       | Low energy: safe action |
| 30-39         | Drill          | Medium energy: risky action possible |
| 40+           | Drill          | High energy: exploit environment |

### Convergence

- **Iterations to Convergence**: ~50 iterations (depending on θ)
- **Convergence Threshold (θ)**: 0.0001
- **Discount Factor (γ)**: 0.9

## Parameters

### Environment Configuration

```python
env = MarsRoverEnv(
    storm_prob=0.2,  # 20% chance of harvest failure
    gamma=0.9        # Discount factor (future rewards less important)
)
```

### Algorithm Parameters

```python
V, deltas = value_iteration(
    env,
    theta=0.0001  # Convergence threshold
)
```

## Performance Metrics

After running 10 episodes with the optimal policy:

- **Average Reward**: Positive (typically 5-15)
- **Successful Episodes**: >80% (reach positive terminal state)
- **Max Reward**: Achievable by harvesting and drilling efficiently
- **Min Reward**: Possible when storms interrupt harvesting

## Mathematical Background

### Stochastic Transitions

The Harvest action demonstrates stochasticity:

$$P(s' | s, \text{Harvest}) = \begin{cases}
0.8 \to \min(100, s+20) & \text{(clear weather)} \\
0.2 \to s & \text{(dust storm)}
\end{cases}$$

### Q-Value Calculation

For example, calculating Q(s=10, a=Drill) manually:

- Success (90%): +10 reward, transition to state 0 (V(0)=0)
- Failure (10%): -1 penalty, stay at state 10

$$Q(10, \text{Drill}) = 0.9 \times [10 + 0.9 \times 0] + 0.1 \times [-1 + 0.9 \times V(10)]$$

## Files and Functions

### `mars_rover_mdp.py`

- **`MarsRoverEnv`**: Environment class
  - `get_transition_reward()`: Returns (prob, next_state, reward) tuples
  
- **`value_iteration()`**: Computes optimal value function
  
- **`extract_policy()`**: Derives greedy policy from value function
  
- **`run_simulation()`**: Executes one episode following a policy
  
- **`print_value_function()`**: Displays V(s) in tabular format
  
- **`print_policy()`**: Displays π*(s) in tabular format

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

See `requirements.txt` for exact versions.

## Learning Outcomes

After studying this project, you should understand:

✅ **Markov Decision Processes**: States, actions, transitions, rewards

✅ **Bellman Equations**: Expectation and Optimality equations

✅ **Value Iteration**: Dynamic programming for MDP solving

✅ **Stochastic Environments**: Handling probabilistic transitions

✅ **Policy Extraction**: Converting value functions to deterministic policies

✅ **Convergence Analysis**: Monitoring and verifying algorithm convergence

## Interesting Findings

🔍 **Fun Facts:**

1. **Dust storms create a dilemma**: Even though Harvest has a 20% failure rate, it's often worth attempting because the energy boost is significant.

2. **Risk tolerance increases with energy**: When battery is high, the rover takes riskier actions (Drill) because it can afford to fail.

3. **Energy thresholds matter**: There are critical energy levels where the optimal action switches (e.g., from Transmit to Drill).

4. **Discount factor effects**: A smaller γ makes the rover more impatient and less concerned about long-term survival.

## References

- Sutton, R. S., & Barto, A. G. (2018). **Reinforcement Learning: An Introduction**. MIT Press.
- Puterman, M. L. (1994). **Markov Decision Processes: Discrete Stochastic Dynamic Programming**. Wiley.
- Russell, S. J., & Norvig, P. (2020). **Artificial Intelligence: A Modern Approach** (4th ed.). Pearson.

## Author

**Mobin Ghorbani** - CS Student @ University of Tehran

- GitHub: [@VictimPickle](https://github.com/VictimPickle)
- Location: Tehran, Iran

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Course**: Artificial Intelligence (University of Tehran)
- **Instructor**: Faculty guidance on MDP theory and implementation
- **References**: Sutton & Barto, Russell & Norvig

---

**Last Updated**: December 24, 2025

⭐ If you found this helpful, please consider starring the repository!
