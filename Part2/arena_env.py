"""
Base Gym environment wrapper for the arena.
Provides reset(), step(), render() API for reinforcement learning.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, MAX_EPISODE_STEPS, FRAMES_PER_SECOND,
    PLAYER_MAX_HEALTH, POSITION_MAX, VELOCITY_MAX, DISTANCE_MAX
)
from arena import Arena


class ArenaEnv(gym.Env):
    """
    Base Gymnasium environment for the arena game.
    Subclasses implement specific control schemes.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FRAMES_PER_SECOND}
    
    def __init__(self, render_mode=None):
        """Initialize environment with observation and action spaces."""
        super().__init__()
        
        self.render_mode = render_mode
        self.arena = Arena()
        
        # Observation space: fixed-size vector with game state features
        # 21 features total
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(21,), dtype=np.float32
        )
        
        # Action space defined by subclasses
        self.action_space = None
        
        # Pygame setup
        self.screen = None
        self.clock = None
        self.is_pygame_initialized = False
        
        # Episode tracking
        self.current_step = 0
        self.previous_health = PLAYER_MAX_HEALTH
    
    def _get_observation(self):
        """Build observation vector from current game state."""
        player = self.arena.player
        
        # Normalize player position to [-1, 1]
        normalized_player_x = (player.position_x / WINDOW_WIDTH) * 2 - 1
        normalized_player_y = (player.position_y / WINDOW_HEIGHT) * 2 - 1
        
        # Normalize velocity
        normalized_velocity_x = np.clip(player.velocity_x / VELOCITY_MAX, -1, 1)
        normalized_velocity_y = np.clip(player.velocity_y / VELOCITY_MAX, -1, 1)
        
        # Normalize orientation to [-1, 1] (from -180 to 180 degrees)
        normalized_orientation = player.orientation / 180.0
        
        # Normalize health
        normalized_health = player.health / PLAYER_MAX_HEALTH
        
        # Nearest enemy info
        nearest_enemy = self.arena.get_nearest_enemy()
        if nearest_enemy is not None:
            nearest_enemy_distance, nearest_enemy_direction_x, nearest_enemy_direction_y = nearest_enemy
            normalized_enemy_distance = np.clip(nearest_enemy_distance / DISTANCE_MAX, 0, 1)
        else:
            normalized_enemy_distance = 1.0  # max distance if no enemies
            nearest_enemy_direction_x = 0.0
            nearest_enemy_direction_y = 0.0
        
        # Nearest spawner info
        nearest_spawner = self.arena.get_nearest_spawner()
        if nearest_spawner is not None:
            nearest_spawner_distance, nearest_spawner_direction_x, nearest_spawner_direction_y = nearest_spawner
            normalized_spawner_distance = np.clip(nearest_spawner_distance / DISTANCE_MAX, 0, 1)
        else:
            normalized_spawner_distance = 1.0
            nearest_spawner_direction_x = 0.0
            nearest_spawner_direction_y = 0.0
        
        # Fire cooldown info (helps agent know when to shoot)
        from config import PLAYER_FIRE_COOLDOWN
        normalized_fire_cooldown = player.fire_cooldown / PLAYER_FIRE_COOLDOWN
        can_shoot = 1.0 if player.fire_cooldown == 0 else 0.0
        
        # Border proximity
        border_margin = 80.0
        dist_to_left = min(player.position_x / border_margin, 1.0)
        dist_to_right = min((WINDOW_WIDTH - player.position_x) / border_margin, 1.0)
        dist_to_top = min(player.position_y / border_margin, 1.0)
        dist_to_bottom = min((WINDOW_HEIGHT - player.position_y) / border_margin, 1.0)
        min_border_dist_x = min(dist_to_left, dist_to_right)
        min_border_dist_y = min(dist_to_top, dist_to_bottom)
        
        # Direction to center
        center_x = WINDOW_WIDTH / 2
        center_y = WINDOW_HEIGHT / 2
        to_center_x = center_x - player.position_x
        to_center_y = center_y - player.position_y
        dist_to_center = max((to_center_x**2 + to_center_y**2)**0.5, 1.0)
        dir_to_center_x = to_center_x / dist_to_center
        dir_to_center_y = to_center_y / dist_to_center
        
        # Counts (normalized)
        normalized_enemy_count = min(self.arena.get_enemy_count() / 10.0, 1.0)
        normalized_spawner_count = min(self.arena.get_spawner_count() / 6.0, 1.0)
        normalized_phase = self.arena.current_phase / 5.0
        
        observation = np.array([
            normalized_player_x,
            normalized_player_y,
            normalized_velocity_x,
            normalized_velocity_y,
            normalized_orientation,
            normalized_health,
            normalized_fire_cooldown,
            can_shoot,
            normalized_enemy_distance,
            nearest_enemy_direction_x,
            nearest_enemy_direction_y,
            normalized_spawner_distance,
            nearest_spawner_direction_x,
            nearest_spawner_direction_y,
            normalized_enemy_count,
            normalized_spawner_count,
            normalized_phase,
            min_border_dist_x,
            min_border_dist_y,
            dir_to_center_x,
            dir_to_center_y
        ], dtype=np.float32)
        
        return observation
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.arena.reset()
        self.current_step = 0
        self.previous_health = PLAYER_MAX_HEALTH
        self._last_projectile_count = 0
        
        observation = self._get_observation()
        info = {"phase": self.arena.current_phase}
        
        return observation, info
    
    def step(self, action):
        """Execute action and return transition."""
        self.current_step += 1
        
        # Track previous state
        prev_enemies_killed = self.arena.total_enemies_killed
        prev_spawners_killed = self.arena.total_spawners_killed
        prev_nearest_enemy = self.arena.get_nearest_enemy()
        prev_nearest_spawner = self.arena.get_nearest_spawner()
        
        # Apply action through subclass implementation
        def action_callback(player):
            self._apply_action(player, action)
        
        # Execute arena step
        result = self.arena.step(action_callback)
        
        # Get observation
        observation = self._get_observation()
        
        reward = result["reward"]
        reward -= 0.01
        
        current_nearest_spawner = self.arena.get_nearest_spawner()
        current_nearest_enemy = self.arena.get_nearest_enemy()
        player = self.arena.player
        speed = (player.velocity_x**2 + player.velocity_y**2)**0.5
        is_moving = speed > 1.5
        
        # Approach spawner
        if current_nearest_spawner is not None and prev_nearest_spawner is not None:
            distance_reduced = prev_nearest_spawner[0] - current_nearest_spawner[0]
            reward += distance_reduced * 1.0
        
        # Stationary penalty
        if not is_moving:
            reward -= 0.1
        
        # Enemy proximity penalty
        if current_nearest_enemy is not None:
            enemy_dist = current_nearest_enemy[0]
            if enemy_dist < 80:
                danger = (80 - enemy_dist) / 80.0
                reward -= danger * 0.3
        
        # Spawner hit bonus (requires movement)
        spawner_hits = result.get("spawner_hits", 0)
        if spawner_hits > 0 and is_moving:
            reward += 15.0
            
        # Kill bonuses
        if self.arena.total_spawners_killed > prev_spawners_killed:
            reward += 100.0
        if self.arena.total_enemies_killed > prev_enemies_killed:
            reward += 5.0
        
        # Termination
        terminated = result["done"]
        truncated = self.current_step >= MAX_EPISODE_STEPS
        
        info = {
            "phase": result["current_phase"],
            "enemies_killed": result["enemies_killed"],
            "spawners_killed": result["spawners_killed"],
            "phase_completed": result["phase_completed"],
            "hits_dealt": result.get("hits_dealt", 0)
        }
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, player, action):
        """Apply action to player. Implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _apply_action")
    
    def render(self):
        """Render current game state."""
        if self.render_mode is None:
            return None
        
        if not self.is_pygame_initialized:
            pygame.init()
            pygame.display.set_caption("Deep RL Arena")
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()
            self.is_pygame_initialized = True
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None
        
        # Draw arena
        self.arena.draw(self.screen)
        
        pygame.display.flip()
        self.clock.tick(FRAMES_PER_SECOND)
        
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self.is_pygame_initialized:
            pygame.quit()
            self.is_pygame_initialized = False
            self.screen = None
            self.clock = None
