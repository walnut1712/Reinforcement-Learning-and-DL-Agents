"""
Q-Learning implementation with epsilon-greedy exploration.
Off-policy temporal difference learning algorithm.
Supports optional intrinsic reward for exploration bonus.
"""

import math
import random
from collections import defaultdict
from config import (
    TRAINING_EPISODES, LEARNING_RATE, DISCOUNT_FACTOR,
    EPSILON_START, EPSILON_END, EPSILON_DECAY_EPISODES,
    INTRINSIC_REWARD_STRENGTH
)
from gridworld import ACTIONS


class QLearningAgent:
    """
    Q-Learning agent using epsilon-greedy policy.
    Updates Q-values using off-policy max-Q update rule.
    """
    
    def __init__(self, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
                 epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                 epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
                 use_intrinsic_reward=False,
                 intrinsic_reward_strength=INTRINSIC_REWARD_STRENGTH):
        """
        Initialize Q-Learning agent with specified parameters.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.use_intrinsic_reward = use_intrinsic_reward
        self.intrinsic_reward_strength = intrinsic_reward_strength
        
        # Q-table maps (state, action) pairs to Q-values
        self.q_table = defaultdict(float)
        
        # Current exploration rate
        self.epsilon = epsilon_start
        
        # Training statistics storage
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Per-episode state visit counter for intrinsic reward
        self.state_visit_counts = defaultdict(int)
    
    def get_epsilon(self, episode_number):
        """
        Calculate epsilon using linear decay schedule.
        Returns epsilon value for given episode number.
        """
        if episode_number >= self.epsilon_decay_episodes:
            return self.epsilon_end
        
        # Linear interpolation from start to end
        decay_progress = episode_number / self.epsilon_decay_episodes
        epsilon_value = self.epsilon_start - decay_progress * (self.epsilon_start - self.epsilon_end)
        
        return epsilon_value
    
    def get_q_value(self, state, action):
        """
        Retrieve Q-value for given state-action pair.
        Returns 0.0 for unvisited pairs due to defaultdict.
        """
        return self.q_table[(state, action)]
    
    def get_max_q_value(self, state):
        """
        Find maximum Q-value across all actions for given state.
        Used in Q-learning update for bootstrapping.
        """
        q_values = [self.get_q_value(state, action) for action in ACTIONS]
        return max(q_values)
    
    def calculate_intrinsic_reward(self, state):
        """
        Calculate intrinsic exploration bonus.
        Formula: intrinsic_reward_strength / sqrt(visit_count + 1)
        Visit count is tracked per episode.
        """
        visit_count = self.state_visit_counts[state]
        intrinsic_reward = self.intrinsic_reward_strength / math.sqrt(visit_count + 1)
        return intrinsic_reward
    
    def reset_episode_visit_counts(self):
        """Reset state visit counter at start of each episode."""
        self.state_visit_counts = defaultdict(int)
    
    def select_action(self, state, epsilon=None):
        """
        Select action using epsilon-greedy policy.
        Implements random tie-breaking when multiple actions share max Q-value.
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Exploration: choose random action
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        
        # Exploitation: choose best action with random tie-breaking
        q_values = [self.get_q_value(state, action) for action in ACTIONS]
        max_q_value = max(q_values)
        
        # Collect all actions with maximum Q-value
        best_actions = [action for action, q_value in zip(ACTIONS, q_values) 
                       if q_value == max_q_value]
        
        # Random selection among best actions
        return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, done):
        """
        Apply Q-learning update rule.
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        Uses maximum Q-value of next state (off-policy).
        """
        current_q_value = self.get_q_value(state, action)
        
        if done:
            # Terminal state has no future value
            target_q_value = reward
        else:
            # Bootstrap using max Q-value of next state
            max_next_q_value = self.get_max_q_value(next_state)
            target_q_value = reward + self.discount_factor * max_next_q_value
        
        # Compute temporal difference error and update
        temporal_difference_error = target_q_value - current_q_value
        new_q_value = current_q_value + self.learning_rate * temporal_difference_error
        
        self.q_table[(state, action)] = new_q_value
    
    def train_episode(self, environment, episode_number, render=False, render_delay=0):
        """
        Execute one training episode.
        Returns tuple of (total_reward, step_count, success_flag).
        """
        import pygame
        
        state = environment.reset()
        self.epsilon = self.get_epsilon(episode_number)
        
        # Reset per-episode visit counts for intrinsic reward
        if self.use_intrinsic_reward:
            self.reset_episode_visit_counts()
        
        total_reward = 0.0
        step_count = 0
        maximum_steps = 500  # Prevent infinite episode loops
        
        while step_count < maximum_steps:
            # Select and execute action in environment
            action = self.select_action(state, self.epsilon)
            next_state, environment_reward, done, info = environment.step(action)
            
            # Compute combined reward with optional intrinsic component
            if self.use_intrinsic_reward:
                # Increment visit count for next_state before calculating intrinsic reward
                self.state_visit_counts[next_state] += 1
                intrinsic_reward = self.calculate_intrinsic_reward(next_state)
                combined_reward = environment_reward + intrinsic_reward
            else:
                combined_reward = environment_reward
            
            # Apply Q-learning update with combined reward
            self.update(state, action, combined_reward, next_state, done)
            
            # Track only environment reward for statistics
            total_reward += environment_reward
            step_count += 1
            state = next_state
            
            # Optional visualization during training
            if render:
                algorithm_label = "Q-Learning (Intrinsic)" if self.use_intrinsic_reward else "Q-Learning"
                should_continue = environment.render(
                    episode_number=episode_number,
                    epsilon=self.epsilon,
                    algorithm_name=algorithm_label
                )
                if not should_continue:
                    return total_reward, step_count, False
                
                if render_delay > 0:
                    pygame.time.wait(render_delay)
            
            if done:
                break
        
        # Store episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(step_count)
        
        success_flag = done and not environment.is_dead
        return total_reward, step_count, success_flag
    
    def get_greedy_action(self, state):
        """
        Select best action without exploration.
        Used during policy demonstration.
        """
        return self.select_action(state, epsilon=0.0)
    
    def demonstrate(self, environment, delay_milliseconds=200):
        """
        Run demonstration using learned greedy policy.
        Returns tuple of (total_reward, success_flag).
        """
        import pygame
        
        state = environment.reset()
        total_reward = 0.0
        maximum_steps = 200
        
        for step_index in range(maximum_steps):
            # Display current state
            should_continue = environment.render(
                algorithm_name="Q-Learning Demo"
            )
            if not should_continue:
                return total_reward, False
            
            pygame.time.wait(delay_milliseconds)
            
            # Execute greedy action
            action = self.get_greedy_action(state)
            next_state, reward, done, info = environment.step(action)
            
            total_reward += reward
            state = next_state
            
            if done:
                # Show final state
                environment.render(algorithm_name="Q-Learning Demo")
                pygame.time.wait(delay_milliseconds * 3)
                break
        
        success_flag = environment.is_done and not environment.is_dead
        return total_reward, success_flag
    
    def get_policy(self):
        """
        Extract learned policy from Q-table.
        Returns dictionary mapping states to best actions.
        """
        policy = {}
        states_seen = set()
        
        # Collect all visited states
        for (state, action) in self.q_table.keys():
            states_seen.add(state)
        
        # Map each state to its best action
        for state in states_seen:
            policy[state] = self.get_greedy_action(state)
        
        return policy
    
    def get_statistics(self):
        """
        Retrieve training statistics.
        Returns dictionary with episode counts and averages.
        """
        if not self.episode_rewards:
            return {"episodes": 0}
        
        return {
            "episodes": len(self.episode_rewards),
            "total_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "average_reward_last_100": sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards)),
            "average_length_last_100": sum(self.episode_lengths[-100:]) / min(100, len(self.episode_lengths)),
        }
