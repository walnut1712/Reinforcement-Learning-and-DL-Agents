# Part 2: Deep Reinforcement Learning Arena

A real-time Pygame arena game where a deep RL agent learns to control a player ship, destroy enemies, and eliminate spawners across multiple phases.

## Project Structure

```
Part2/
├── config.py           # Game settings and hyperparameters
├── arena.py            # Core game logic (Player, Enemy, Spawner, Projectile)
├── arena_env.py        # Base Gymnasium environment wrapper
├── rotation_env.py     # Rotation-based control scheme (thrust/rotate)
├── directional_env.py  # Directional control scheme (WASD-style)
├── train_rotation.py   # Training script for rotation control
├── train_directional.py# Training script for directional control
├── evaluate.py         # Visual evaluation of trained agents
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

```bash
cd Part2
pip install -r requirements.txt
```

## Quick Start

### 1. Test the Environment (Random Actions)
```bash
python evaluate.py --random --episodes 1
```

### 2. Train the Agents

Training automatically launches TensorBoard and opens your browser to view training curves.

```bash
# Train rotation control agent (thrust/rotate like a spaceship)
python train_rotation.py --timesteps 500000

# Train directional control agent (WASD-style movement)
python train_directional.py --timesteps 500000
```

To disable automatic TensorBoard launch:
```bash
python train_rotation.py --timesteps 500000 --no-tensorboard
```

### 3. Monitor Training with TensorBoard

TensorBoard is **automatically launched** when training starts. If you need to run it manually:

```bash
cd Part2
tensorboard --logdir=logs
```

Then open browser at http://localhost:6006

### 4. Evaluate Trained Agents
```bash
python evaluate.py --model rotation --episodes 5
python evaluate.py --model directional --episodes 5
```

## Control Schemes

Both control schemes use **MultiDiscrete** action spaces, allowing simultaneous movement and shooting.

### Rotation Control (MultiDiscrete [2, 3, 2])
| Channel | Options |
|---------|---------|
| 1. Thrust | 0: None, 1: Thrust |
| 2. Rotate | 0: None, 1: Left, 2: Right |
| 3. Shoot | 0: None, 1: Shoot |

Allows simultaneous **Forward + Left/Right + Fire**.

### Directional Control (MultiDiscrete [5, 2])
| Movement Action | Description |
|-----------------|-------------|
| 0 | No movement (stop) |
| 1 | Move up |
| 2 | Move down |
| 3 | Move left |
| 4 | Move right |

| Shoot Action | Description |
|--------------|-------------|
| 0 | Don't shoot |
| 1 | Shoot |

## Observation Vector

The agent receives a 21-dimensional normalized feature vector:

| Feature | Description |
|---------|-------------|
| player_x, player_y | Normalized player position |
| velocity_x, velocity_y | Normalized player velocity |
| orientation | Player facing direction |
| health | Normalized player health |
| fire_cooldown | Normalized cooldown until next shot |
| can_shoot | 1.0 if can shoot, 0.0 otherwise |
| enemy_distance | Distance to nearest enemy |
| enemy_direction_x, enemy_direction_y | Direction to nearest enemy |
| spawner_distance | Distance to nearest spawner |
| spawner_direction_x, spawner_direction_y | Direction to nearest spawner |
| enemy_count | Number of active enemies |
| spawner_count | Number of active spawners |
| current_phase | Current game phase |
| min_border_dist_x | Distance to nearest horizontal edge |
| min_border_dist_y | Distance to nearest vertical edge |
| dir_to_center_x, dir_to_center_y | Direction to arena center |

## Reward Structure

| Event | Reward | Justification |
|-------|--------|---------------|
| Destroy enemy | +10 | Encourages combat engagement |
| Destroy spawner | +50 | Primary objective to progress |
| Complete phase | +100 | Major milestone reward |
| Take damage | -5 | Discourages reckless behavior |
| Death | -100 | Strong penalty for failing |
| Time step | -0.01 | Encourages efficient play |
| Approach spawner (shaping) | +0.01 | Guides agent toward objectives when safe |

## Command Line Options

### Training Scripts
| Option | Default | Description |
|--------|---------|-------------|
| `--timesteps` | 500000 | Total training timesteps |
| `--learning-rate` | 0.0003 | PPO learning rate |
| `--save-freq` | 50000 | Checkpoint save frequency |
| `--no-tensorboard` | False | Disable automatic TensorBoard launch |

### Evaluation Script
| Option | Default | Description |
|--------|---------|-------------|
| `--model` | rotation | Model to evaluate (rotation or directional) |
| `--episodes` | 5 | Number of episodes to run |
| `--random` | False | Use random actions instead of trained model |

## Output Files

After training:
```
models/
├── rotation_model.zip           # Final rotation model
├── directional_model.zip        # Final directional model
├── rotation_checkpoint_*.zip    # Periodic checkpoints
└── directional_checkpoint_*.zip # Periodic checkpoints

logs/
├── rotation/     # TensorBoard logs for rotation training
└── directional/  # TensorBoard logs for directional training
```

## Technical Details

- **Window Size**: 960 x 680 pixels
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network Architecture**: MLP with two hidden layers [128, 128]
- **Training**: Headless (no rendering) for speed
- **Framework**: Stable Baselines3 with Gymnasium
- **TensorBoard**: Automatically launched during training
