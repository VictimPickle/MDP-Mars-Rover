# Algorithms and Theoretical Foundation

## Value Iteration Algorithm

### Overview

**Value Iteration** is a dynamic programming algorithm that finds the optimal policy by iteratively improving the value function. Unlike Policy Iteration, it combines policy evaluation and improvement into a single step.

### Algorithm Pseudocode

```
Function VALUE_ITERATION(env, theta=0.0001):
    V ← {s: 0 for all states s}  // Initialize value function
    
    repeat:
        delta ← 0
        
        for each state s in env.states:
            if s is terminal:
                continue
            
            old_v ← V[s]
            
            // Find the action with maximum Q-value
            max_q ← -∞
            for each action a in env.actions:
                q_value ← 0
                for each (prob, next_s, reward) in env.get_transition_reward(s, a):
                    q_value += prob * (reward + gamma * V[next_s])
                max_q ← max(max_q, q_value)
            
            V[s] ← max_q
            delta ← max(delta, |old_v - V[s]|)
        
        // Check convergence
        if delta < theta:
            break
    
    return V
```

### Key Differences from Policy Iteration

| Aspect | Value Iteration | Policy Iteration |
|--------|-----------------|------------------|
| **Approach** | Combined evaluation & improvement | Separate evaluation & improvement |
| **Inner Loop** | Maximize over all actions | Fixed policy evaluation |
| **Convergence** | Generally slower | May be faster when policies stabilize early |
| **Implementation** | Slightly simpler | More structure, easier to debug |
| **Best For** | Stochastic environments (Mars Rover) | Deterministic environments (Gridworld) |

### Complexity Analysis

- **Time Complexity**: O(k * |S| * |A| * |S'|)
  - k = iterations to convergence
  - |S| = number of states
  - |A| = number of actions
  - |S'| = expected number of next states

- **Space Complexity**: O(|S|)
  - Single value function array

### Convergence Guarantees

1. **Guaranteed Convergence**: Value Iteration always converges to the optimal value function V* as long as:
   - The discount factor γ < 1, OR
   - The environment is acyclic (no infinite positive reward loops)

2. **Optimality**: The converged value function is the unique optimal value function.

3. **Threshold (θ)**: Controls precision. Smaller θ means more iterations but more accurate solution.

## Bellman Equations

### Bellman Expectation Equation

The value of a state under policy π is:

$$V_{\pi}(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s,a) \left[ R(s,a,s') + \gamma V_{\pi}(s') \right]$$

**Interpretation**: The value of a state is the expected immediate reward plus the discounted future rewards.

### Bellman Optimality Equation

The optimal value of a state is:

$$V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

Or equivalently:

$$V^*(s) = \max_{a \in A} Q^*(s, a)$$

**Interpretation**: The optimal value is achieved by choosing the action that maximizes the expected return.

### Q-Value

The Q-value (action-value function) is:

$$Q(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V(s') \right]$$

**Interpretation**: The expected return for taking action a in state s and then following the current policy.

## Mars Rover Environment Details

### State Space

$$S = \{0, 10, 20, 30, \ldots, 100\}$$ (battery levels)

### Action Space

$$A = \{\text{Harvest}, \text{Drill}, \text{Transmit}\}$$

### Transitions and Rewards

#### Harvest Action

**Success (probability = 1 - storm_prob = 0.8)**:
- Next state: min(battery + 20, 100)
- Reward: 0

**Failure (probability = storm_prob = 0.2)**:
- Next state: battery (no change)
- Reward: 0

#### Drill Action

**Success (probability = 1.0, if battery ≥ 30)**:
- Next state: battery - 30
- Reward: 10

**Failure (probability = 1.0, if battery < 30)**:
- Next state: battery (no change)
- Reward: -1 (penalty)

#### Transmit Action

**Success (probability = 1.0, if battery ≥ 10)**:
- Next state: battery - 10
- Reward: 5

**Failure (probability = 1.0, if battery < 10)**:
- Next state: battery (no change)
- Reward: -1 (penalty)

### Terminal State

Battery = 0 (game over, no more actions possible)

## Example: Q-Value Calculation

### Problem

Given:
- State s = 10 (current battery)
- Action a = Drill
- γ = 0.9
- Transitions:
  - Success (90%): Reward +10, next state = 0 (terminal)
  - Failure (10%): Reward -1, next state = 10

Assuming:
- V(0) = 0 (terminal state has 0 value)
- V(10) ≈ some value (self-loop)

### Solution

$$Q(10, \text{Drill}) = 0.9 \times [10 + 0.9 \times V(0)] + 0.1 \times [-1 + 0.9 \times V(10)]$$

$$= 0.9 \times [10 + 0] + 0.1 \times [-1 + 0.9 \times V(10)]$$

$$= 9.0 + 0.1 \times [-1 + 0.9 \times V(10)]$$

If V(10) \approx 10 (from convergence):

$$Q(10, \text{Drill}) \approx 9.0 + 0.1 \times [-1 + 9.0] = 9.0 + 0.8 = 9.8$$

## Policy Extraction

### From Value Function

Once the value function V* is computed, the optimal policy is extracted as:

$$\pi^*(s) = \arg\max_{a \in A} \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

In code:

```python
def extract_policy(env, V):
    policy = {}
    for s in env.states:
        if s == 0:
            policy[s] = None  # Terminal state
            continue
        
        best_action = None
        best_q = -float('inf')
        
        for a in env.actions:
            transitions = env.get_transition_reward(s, a)
            q_value = sum(prob * (reward + env.gamma * V[next_s])
                         for prob, next_s, reward in transitions)
            
            if q_value > best_q:
                best_q = q_value
                best_action = a
        
        policy[s] = best_action
    
    return policy
```

## Convergence Behavior

### Delta Sequence

The algorithm monitors how much values change each iteration:

$$\text{delta}_i = \max_{s} |V_i(s) - V_{i-1}(s)|$$

Convergence occurs when:

$$\text{delta}_i < \theta$$

### Typical Convergence Pattern

1. **Early iterations**: Large delta (rapid value changes)
2. **Middle iterations**: Medium delta (values stabilizing)
3. **Final iterations**: Delta < theta (convergence achieved)

For Mars Rover with θ = 0.0001:
- Usually converges in 30-60 iterations
- Delta typically decreases exponentially

## Implementation Tips

### 1. Numerical Stability

```python
# Avoid division by zero
if next_state not in V:
    V[next_state] = 0  # Default for new states
```

### 2. Terminal State Handling

```python
# Terminal states should have V = 0 (no future)
if state == 0:  # Terminal
    V[state] = 0
```

### 3. Convergence Detection

```python
# Use max absolute change as convergence criterion
delta = max(abs(V[s] - new_V[s]) for s in env.states)
if delta < theta:
    break
```

### 4. Discount Factor Sensitivity

- **γ = 0.9**: Prefers immediate rewards (short-term thinking)
- **γ = 0.99**: Balances immediate and future rewards
- **γ = 0.999**: Values long-term consequences highly

## Performance Optimization

### Vectorization

For large state/action spaces, use NumPy:

```python
# Instead of loops:
action_values = np.array([
    sum(prob * (reward + env.gamma * V[next_s])
        for prob, next_s, reward in env.get_transition_reward(s, a))
    for a in env.actions
])
best_action = np.argmax(action_values)
```

### Sparse Representations

For large MDPs, use dictionaries instead of matrices to store only non-zero values.

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
