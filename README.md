# Mars Rover MDP

Implementation of Mars Rover problem using Markov Decision Process (MDP) and Value Iteration algorithm for optimal policy learning.

## Problem Description

A Mars rover must optimize battery management while exploring the Martian surface under stochastic conditions. The rover receives exploration points for successful missions but incurs energy penalties. The challenge is to find the optimal policy that maximizes expected cumulative reward over a finite horizon.

## Algorithm Details

### Value Iteration

Value Iteration is a dynamic programming approach that iteratively computes the optimal value function:

```
V(s) = max_a [R(s,a) + γ * Σ_s' P(s'|s,a) * V(s')]
```

Where:
- `V(s)`: Optimal value of state s
- `R(s,a)`: Reward for action a in state s
- `γ`: Discount factor
- `P(s'|s,a)`: Transition probability

### State Space

- Battery level: [0, 100]% (discrete intervals)
- Location: Grid position (x, y)
- Task completion status: Completed or pending

### Action Space

- **Move**: Navigate to adjacent grid cell (-2 battery)
- **Recharge**: Restore battery to 100% (-1 battery during transition)
- **Scout**: Gather data at current location (+10 reward, -5 battery)
- **Wait**: No action (-0.5 battery decay)

### Reward Function

```
R(state, action) = {
    +10   if successfully scout → task completed
    -2    if move without enough battery → failed
    +0    if wait → survival
    -1    if recharge → energy cost
}
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python MDP_Mars_Rover.py
```

### With Custom Parameters

```python
from mars_rover_mdp import ValueIterationAgent

# Create agent with custom discount factor
agent = ValueIterationAgent(gamma=0.95, max_iterations=1000)

# Compute optimal policy
optimal_policy = agent.value_iteration()

# Get action for current state
state = (50, 5, 5)  # (battery%, x, y)
action = optimal_policy[state]
```

## Output

The program generates:

1. **Optimal value function**: V(s) for all states
2. **Optimal policy**: π*(s) mapping states to best actions
3. **Value convergence plot**: Showing iteration progress
4. **Policy visualization**: Heatmap of action selection across state space
5. **Performance metrics**: Average reward, convergence speed

## Key Results

- Convergence: ~50-100 iterations for typical parameters
- Optimal strategy: Scout aggressively early, recharge strategically
- Performance: Average reward per episode increases with proper policy

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0

See `requirements.txt` for full dependencies.

## Implementation Notes

- Value Iteration guarantees convergence to optimal V-function
- Policy extraction deterministic after convergence
- Stochastic transitions handled via expectation in Bellman update
- Computational complexity: O(|S|² |A|) per iteration

## License

MIT License - see LICENSE file for details

## Author

Mobin Ghorbani

## References

- Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.