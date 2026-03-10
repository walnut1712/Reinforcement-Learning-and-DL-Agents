"""
Configuration parameters for Q-Learning and SARSA training.
All training hyperparameters and environment settings defined here.
"""

# Grid display dimensions
GRID_WIDTH = 10
GRID_HEIGHT = 10
CELL_SIZE = 60

# Training hyperparameters
TRAINING_EPISODES = 1200
LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.95

# Epsilon-greedy exploration schedule (linear decay)
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 400

# Monster movement probability (40 percent chance to move)
MONSTER_MOVE_PROBABILITY = 0.4

# Environment reward values (must not be altered)
REWARD_APPLE = 1.0
REWARD_CHEST = 2.0
REWARD_KEY = 0.0
REWARD_DEATH = -6.0
REWARD_STEP = -0.05
REWARD_WALL_BUMP = -0.01

# Intrinsic reward for exploration bonus
INTRINSIC_REWARD_STRENGTH = 0.1

# Visualization timing
FRAMES_PER_SECOND = 60
ANIMATION_DELAY_TRAINING = 0
ANIMATION_DELAY_DEMO = 200

# Color definitions (RGB format)
COLOR_BACKGROUND = (40, 44, 52)
COLOR_GRID_LINE = (60, 64, 72)
COLOR_AGENT = (97, 175, 239)
COLOR_APPLE = (152, 195, 121)
COLOR_KEY = (229, 192, 123)
COLOR_CHEST = (198, 120, 95)
COLOR_CHEST_OPEN = (139, 84, 66)
COLOR_ROCK = (92, 99, 112)
COLOR_FIRE = (224, 108, 117)
COLOR_MONSTER = (198, 120, 221)
COLOR_FLOOR = (50, 54, 62)
COLOR_TEXT = (171, 178, 191)
