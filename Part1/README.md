# Part 1: Tabular Reinforcement Learning in Gridworld

Implementation of Q-Learning and SARSA algorithms for a Pygame-based gridworld environment with multiple levels of increasing complexity.

## Project Structure

```
Part1/
├── config.py       # Hyperparameters and game settings
├── gridworld.py    # Gridworld environment with Pygame visualization
├── levels.py       # Level definitions (0-6)
├── q_learning.py   # Q-Learning agent implementation
├── sarsa.py        # SARSA agent implementation
├── main.py         # Training and evaluation script
├── requirements.txt
└── README.md
```

## Installation

```bash
cd Part1
pip install -r requirements.txt
```

## Quick Start

Run all levels with both algorithms:
```bash
python main.py
```

This will:
1. Train Q-Learning and SARSA on Levels 0-5
2. Train with/without intrinsic reward comparison on Level 6
3. Show visual demonstrations of learned policies
4. Display training statistics and success rates

## Level Descriptions

| Level | Description | Features |
|-------|-------------|----------|
| 0 | Simple open grid | 4 apples, no obstacles |
| 1 | Fire corridor | Apples behind fire hazards |
| 2 | Multiple apples | Scattered collectibles |
| 3 | Key and chest | Must collect key before opening chest |
| 4 | Single monster | Moving enemy with 40% move probability |
| 5 | Multiple monsters | Multiple moving enemies |
| 6 | Sparse rewards | Tests intrinsic reward exploration |

## Algorithms

### Q-Learning (Off-policy)
- Update rule: `Q(s,a) = Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]`
- Uses maximum Q-value of next state (greedy)
- Learns optimal policy regardless of exploration behavior

### SARSA (On-policy)
- Update rule: `Q(s,a) = Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]`
- Uses actual next action Q-value
- Learns more conservative policies around hazards

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAINING_EPISODES` | 1000 | Episodes per level |
| `LEARNING_RATE` | 0.1 | Learning rate (α) |
| `DISCOUNT_FACTOR` | 0.95 | Discount factor (γ) |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_END` | 0.01 | Final exploration rate |
| `EPSILON_DECAY_EPISODES` | 800 | Episodes for epsilon decay |
| `MONSTER_MOVE_PROBABILITY` | 0.4 | Monster movement chance per step |
| `INTRINSIC_REWARD_STRENGTH` | 0.5 | Exploration bonus strength |

## Reward Structure

| Event | Reward |
|-------|--------|
| Collect apple | +1.0 |
| Open chest | +2.0 |
| Collect key | 0.0 |
| Death (fire/monster) | -10.0 |
| Step penalty | -0.01 |
| Wall bump | -0.05 |

## Intrinsic Reward (Level 6)

Formula: `r_intrinsic = intrinsic_reward_strength / sqrt(visit_count + 1)`

- Visit counter resets each episode
- Encourages exploration of unvisited states
- Combined with environment reward for Q-value updates

## Key Features

### Epsilon-Greedy Exploration
- Linear decay from `EPSILON_START` to `EPSILON_END`
- Random tie-breaking when multiple actions have equal Q-values

### State Representation
- Agent position (row, column)
- Key possession status (boolean)
- Remaining apples (frozenset)
- Remaining chests (frozenset)
- Monster positions (frozenset)

### Monster Behavior
- 40% probability to move after each agent action
- Moves toward random valid adjacent cell
- Contact with monster causes agent death

## Expected Results

| Level | Q-Learning Success | SARSA Success |
|-------|-------------------|---------------|
| 0 | ~98% | ~97% |
| 1 | ~50% | ~35% |
| 2 | ~87% | ~83% |
| 3 | ~90% | ~86% |
| 4 | ~5% | ~6% |
| 5 | ~0% | ~0% |

**Note**: Monster levels (4-5) have very low success rates due to:
- Stochastic monster movement
- Large state space (monster positions included in state)
- Limited training episodes for tabular methods

## Output

The script displays:
- Episode-by-episode training progress
- Average rewards (last 100 episodes)
- Final success rates
- Visual demonstrations with Pygame

Close the Pygame window will poceed to the next demonstration.
