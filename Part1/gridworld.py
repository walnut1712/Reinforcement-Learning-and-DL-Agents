"""
Gridworld environment with Pygame visualization.
Supports agent movement, collectibles, hazards, and monsters.
"""

import pygame
import random
import copy
from config import (
    GRID_WIDTH, GRID_HEIGHT, CELL_SIZE,
    MONSTER_MOVE_PROBABILITY,
    REWARD_APPLE, REWARD_CHEST, REWARD_KEY, REWARD_DEATH, REWARD_STEP, REWARD_WALL_BUMP,
    COLOR_BACKGROUND, COLOR_GRID_LINE, COLOR_AGENT, COLOR_APPLE, COLOR_KEY,
    COLOR_CHEST, COLOR_CHEST_OPEN, COLOR_ROCK, COLOR_FIRE, COLOR_MONSTER,
    COLOR_FLOOR, COLOR_TEXT, FRAMES_PER_SECOND
)
from levels import parse_grid, ALL_LEVELS

# Action definitions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

# Direction deltas for each action (row_delta, column_delta)
ACTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


class GridWorld:
    """
    Gridworld environment with Pygame rendering.
    Handles game state, transitions, and visualization.
    """
    
    def __init__(self, level_index=0, render_enabled=True):
        """
        Initialize gridworld with specified level.
        
        Args:
            level_index: Index of level to load (0-5)
            render_enabled: Whether to enable Pygame rendering
        """
        self.level_index = level_index
        self.render_enabled = render_enabled
        self.level_data = parse_grid(ALL_LEVELS[level_index])
        
        self.grid_width = self.level_data["width"]
        self.grid_height = self.level_data["height"]
        
        # Pygame initialization
        self.screen = None
        self.clock = None
        self.font = None
        
        if self.render_enabled:
            pygame.init()
            screen_width = self.grid_width * CELL_SIZE
            screen_height = self.grid_height * CELL_SIZE + 60  # Extra space for info
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption(f"Gridworld - {self.level_data['name']}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        self.reset()
    
    def reset(self):
        """
        Reset environment to initial state.
        
        Returns:
            Initial state tuple
        """
        # Agent position
        self.agent_position = self.level_data["start_position"]
        
        # Collectible states (copy to allow modification)
        self.remaining_apples = set(self.level_data["apples"])
        self.remaining_keys = set(self.level_data["keys"])
        self.remaining_chests = set(self.level_data["chests"])
        
        # Monster positions (copy to allow movement)
        self.monster_positions = set(self.level_data["monsters"])
        
        # Agent inventory
        self.has_key = False
        
        # Episode state
        self.is_done = False
        self.is_dead = False
        self.total_reward = 0.0
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation.
        State includes agent position, key status, remaining collectibles, and monster positions.
        
        Returns:
            Hashable state tuple
        """
        # Determine monster proximity directions
        monster_is_up = False
        monster_is_down = False
        monster_is_left = False
        monster_is_right = False
        
        agent_row, agent_column = self.agent_position
        
        for monster_row, monster_column in self.monster_positions:
            distance = abs(agent_row - monster_row) + abs(agent_column - monster_column)
            
            if distance <= 3:
                if monster_row < agent_row:
                    monster_is_up = True
                if monster_row > agent_row:
                    monster_is_down = True
                if monster_column < agent_column:
                    monster_is_left = True
                if monster_column > agent_column:
                    monster_is_right = True
                    
        # State: (agent_row, agent_col, has_key, remaining_apples, remaining_chests, 
        #         monster_up, monster_down, monster_left, monster_right)
        return (
            self.agent_position[0],
            self.agent_position[1],
            self.has_key,
            frozenset(self.remaining_apples),
            frozenset(self.remaining_chests),
            monster_is_up,
            monster_is_down,
            monster_is_left,
            monster_is_right
        )
    
    def _is_valid_position(self, position):
        """
        Check if position is within grid bounds and not blocked.
        
        Args:
            position: Tuple (row, col)
            
        Returns:
            True if position is valid for movement
        """
        row, column = position
        
        # Check bounds
        if row < 0 or row >= self.grid_height:
            return False
        if column < 0 or column >= self.grid_width:
            return False
        
        # Check rocks
        if position in self.level_data["rocks"]:
            return False
        
        return True
    
    def _move_monsters(self):
        """
        Move monsters probabilistically.
        Each monster has MONSTER_MOVE_PROBABILITY chance to move.
        """
        new_monster_positions = set()
        
        for monster_position in self.monster_positions:
            if random.random() < MONSTER_MOVE_PROBABILITY:
                # Get valid adjacent positions
                valid_moves = []
                for action in ACTIONS:
                    delta = ACTION_DELTAS[action]
                    new_position = (
                        monster_position[0] + delta[0],
                        monster_position[1] + delta[1]
                    )
                    
                    # Check if move is valid (not into rocks, fire, or other monsters)
                    if self._is_valid_position(new_position):
                        if new_position not in self.level_data["fire"]:
                            if new_position not in new_monster_positions:
                                valid_moves.append(new_position)
                
                if valid_moves:
                    new_position = random.choice(valid_moves)
                    new_monster_positions.add(new_position)
                else:
                    new_monster_positions.add(monster_position)
            else:
                new_monster_positions.add(monster_position)
        
        self.monster_positions = new_monster_positions
    
    def step(self, action):
        """
        Execute action and return transition.
        
        Args:
            action: Action index (0-3)
            
        Returns:
            Tuple (next_state, reward, done, info)
        """
        if self.is_done:
            return self._get_state(), 0.0, True, {"already_done": True}
        
        reward = REWARD_STEP
        info = {"action": ACTION_NAMES[action]}
        
        # Calculate new position
        delta = ACTION_DELTAS[action]
        new_position = (
            self.agent_position[0] + delta[0],
            self.agent_position[1] + delta[1]
        )
        
        # Check if move is valid
        if self._is_valid_position(new_position):
            self.agent_position = new_position
        else:
            reward += REWARD_WALL_BUMP
            info["wall_bump"] = True
        
        # Check for death by fire
        if self.agent_position in self.level_data["fire"]:
            reward = REWARD_DEATH
            self.is_done = True
            self.is_dead = True
            info["death"] = "fire"
        
        # Check for death by monster
        if self.agent_position in self.monster_positions:
            reward = REWARD_DEATH
            self.is_done = True
            self.is_dead = True
            info["death"] = "monster"
        
        # Collect apple
        if self.agent_position in self.remaining_apples:
            self.remaining_apples.remove(self.agent_position)
            reward += REWARD_APPLE
            info["collected"] = "apple"
        
        # Collect key
        if self.agent_position in self.remaining_keys:
            self.remaining_keys.remove(self.agent_position)
            self.has_key = True
            reward += REWARD_KEY
            info["collected"] = "key"
        
        # Open chest
        if self.agent_position in self.remaining_chests and self.has_key:
            self.remaining_chests.remove(self.agent_position)
            self.has_key = False  # Key is consumed
            reward += REWARD_CHEST
            info["collected"] = "chest"
        
        # Move monsters after agent action
        if not self.is_done:
            self._move_monsters()
            
            # Check for death by monster after monster movement
            if self.agent_position in self.monster_positions:
                reward = REWARD_DEATH
                self.is_done = True
                self.is_dead = True
                info["death"] = "monster_moved"
        
        # Check win condition
        if not self.is_done:
            all_apples_collected = len(self.remaining_apples) == 0
            all_chests_opened = len(self.remaining_chests) == 0
            
            if all_apples_collected and all_chests_opened:
                self.is_done = True
                info["win"] = True
        
        self.total_reward += reward
        self.step_count += 1
        
        return self._get_state(), reward, self.is_done, info
    
    def render(self, episode_number=None, epsilon=None, algorithm_name=None):
        """
        Render current state using Pygame.
        
        Args:
            episode_number: Current episode number for display
            epsilon: Current epsilon value for display
            algorithm_name: Name of algorithm being used
        """
        if not self.render_enabled:
            return
        
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        # Clear screen
        self.screen.fill(COLOR_BACKGROUND)
        
        # Draw grid cells
        for row in range(self.grid_height):
            for column in range(self.grid_width):
                cell_x = column * CELL_SIZE
                cell_y = row * CELL_SIZE
                position = (row, column)
                
                # Draw floor
                pygame.draw.rect(
                    self.screen, COLOR_FLOOR,
                    (cell_x + 1, cell_y + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                )
                
                # Draw rocks
                if position in self.level_data["rocks"]:
                    self._draw_rock(cell_x, cell_y)
                
                # Draw fire
                if position in self.level_data["fire"]:
                    self._draw_fire(cell_x, cell_y)
                
                # Draw collectibles
                if position in self.remaining_apples:
                    self._draw_apple(cell_x, cell_y)
                
                if position in self.remaining_keys:
                    self._draw_key(cell_x, cell_y)
                
                if position in self.remaining_chests:
                    self._draw_chest(cell_x, cell_y, closed=True)
                
                # Draw opened chests (chests that were collected)
                if position in self.level_data["chests"] and position not in self.remaining_chests:
                    self._draw_chest(cell_x, cell_y, closed=False)
                
                # Draw monsters
                if position in self.monster_positions:
                    self._draw_monster(cell_x, cell_y)
        
        # Draw agent
        agent_x = self.agent_position[1] * CELL_SIZE
        agent_y = self.agent_position[0] * CELL_SIZE
        self._draw_agent(agent_x, agent_y)
        
        # Draw grid lines
        for row in range(self.grid_height + 1):
            pygame.draw.line(
                self.screen, COLOR_GRID_LINE,
                (0, row * CELL_SIZE),
                (self.grid_width * CELL_SIZE, row * CELL_SIZE)
            )
        for column in range(self.grid_width + 1):
            pygame.draw.line(
                self.screen, COLOR_GRID_LINE,
                (column * CELL_SIZE, 0),
                (column * CELL_SIZE, self.grid_height * CELL_SIZE)
            )
        
        # Draw info panel
        info_y = self.grid_height * CELL_SIZE + 5
        
        # Level name
        level_text = self.font.render(self.level_data["name"], True, COLOR_TEXT)
        self.screen.blit(level_text, (10, info_y))
        
        # Episode and epsilon
        if episode_number is not None:
            episode_text = self.font.render(f"Episode: {episode_number}", True, COLOR_TEXT)
            self.screen.blit(episode_text, (200, info_y))
        
        if epsilon is not None:
            epsilon_text = self.font.render(f"Epsilon: {epsilon:.3f}", True, COLOR_TEXT)
            self.screen.blit(epsilon_text, (350, info_y))
        
        if algorithm_name is not None:
            algo_text = self.font.render(algorithm_name, True, COLOR_TEXT)
            self.screen.blit(algo_text, (450, info_y))
        
        # Reward and key status
        reward_text = self.font.render(f"Reward: {self.total_reward:.1f}", True, COLOR_TEXT)
        self.screen.blit(reward_text, (10, info_y + 25))
        
        key_status = "Key: Yes" if self.has_key else "Key: No"
        key_text = self.font.render(key_status, True, COLOR_TEXT)
        self.screen.blit(key_text, (150, info_y + 25))
        
        if self.is_dead:
            death_text = self.font.render("DEAD!", True, COLOR_FIRE)
            self.screen.blit(death_text, (250, info_y + 25))
        elif self.is_done:
            win_text = self.font.render("WIN!", True, COLOR_APPLE)
            self.screen.blit(win_text, (250, info_y + 25))
        
        pygame.display.flip()
        self.clock.tick(FRAMES_PER_SECOND)
        
        return True
    
    def _draw_agent(self, cell_x, cell_y):
        """Draw agent as a circle."""
        center_x = cell_x + CELL_SIZE // 2
        center_y = cell_y + CELL_SIZE // 2
        radius = CELL_SIZE // 3
        pygame.draw.circle(self.screen, COLOR_AGENT, (center_x, center_y), radius)
        # Draw eyes
        eye_offset = radius // 3
        eye_radius = 4
        pygame.draw.circle(self.screen, COLOR_BACKGROUND, 
                          (center_x - eye_offset, center_y - eye_offset // 2), eye_radius)
        pygame.draw.circle(self.screen, COLOR_BACKGROUND, 
                          (center_x + eye_offset, center_y - eye_offset // 2), eye_radius)
    
    def _draw_rock(self, cell_x, cell_y):
        """Draw rock as a filled rectangle."""
        margin = 3
        pygame.draw.rect(
            self.screen, COLOR_ROCK,
            (cell_x + margin, cell_y + margin, CELL_SIZE - 2 * margin, CELL_SIZE - 2 * margin)
        )
    
    def _draw_fire(self, cell_x, cell_y):
        """Draw fire as triangles."""
        center_x = cell_x + CELL_SIZE // 2
        bottom_y = cell_y + CELL_SIZE - 8
        
        # Main flame
        points = [
            (center_x, cell_y + 8),
            (center_x - 15, bottom_y),
            (center_x + 15, bottom_y)
        ]
        pygame.draw.polygon(self.screen, COLOR_FIRE, points)
        
        # Inner flame (lighter)
        inner_points = [
            (center_x, cell_y + 18),
            (center_x - 8, bottom_y - 5),
            (center_x + 8, bottom_y - 5)
        ]
        pygame.draw.polygon(self.screen, (255, 180, 100), inner_points)
    
    def _draw_apple(self, cell_x, cell_y):
        """Draw apple as a circle with stem."""
        center_x = cell_x + CELL_SIZE // 2
        center_y = cell_y + CELL_SIZE // 2 + 3
        radius = CELL_SIZE // 4
        pygame.draw.circle(self.screen, COLOR_APPLE, (center_x, center_y), radius)
        # Stem
        pygame.draw.line(self.screen, (139, 90, 43),
                        (center_x, center_y - radius),
                        (center_x + 3, center_y - radius - 8), 2)
    
    def _draw_key(self, cell_x, cell_y):
        """Draw key shape."""
        center_x = cell_x + CELL_SIZE // 2
        center_y = cell_y + CELL_SIZE // 2
        
        # Key handle (circle)
        pygame.draw.circle(self.screen, COLOR_KEY, (center_x - 8, center_y), 10, 3)
        
        # Key shaft
        pygame.draw.rect(self.screen, COLOR_KEY,
                        (center_x - 2, center_y - 3, 20, 6))
        
        # Key teeth
        pygame.draw.rect(self.screen, COLOR_KEY,
                        (center_x + 12, center_y + 3, 4, 8))
    
    def _draw_chest(self, cell_x, cell_y, closed=True):
        """Draw treasure chest."""
        color = COLOR_CHEST if closed else COLOR_CHEST_OPEN
        
        # Chest body
        chest_x = cell_x + 8
        chest_y = cell_y + CELL_SIZE // 2
        chest_width = CELL_SIZE - 16
        chest_height = CELL_SIZE // 2 - 8
        pygame.draw.rect(self.screen, color,
                        (chest_x, chest_y, chest_width, chest_height))
        
        # Chest lid
        lid_height = 12
        if closed:
            pygame.draw.rect(self.screen, color,
                            (chest_x, chest_y - lid_height, chest_width, lid_height))
        else:
            # Open lid (tilted)
            points = [
                (chest_x, chest_y),
                (chest_x + chest_width, chest_y),
                (chest_x + chest_width - 5, chest_y - lid_height - 5),
                (chest_x + 5, chest_y - lid_height - 5)
            ]
            pygame.draw.polygon(self.screen, color, points)
        
        # Lock
        lock_x = cell_x + CELL_SIZE // 2 - 4
        lock_y = chest_y + chest_height // 2 - 4
        pygame.draw.rect(self.screen, COLOR_KEY if closed else COLOR_CHEST_OPEN,
                        (lock_x, lock_y, 8, 8))
    
    def _draw_monster(self, cell_x, cell_y):
        """Draw monster as a spiky shape."""
        center_x = cell_x + CELL_SIZE // 2
        center_y = cell_y + CELL_SIZE // 2
        
        # Monster body
        pygame.draw.circle(self.screen, COLOR_MONSTER, (center_x, center_y), CELL_SIZE // 3)
        
        # Spikes
        spike_length = 8
        for angle_offset in range(0, 360, 45):
            import math
            angle_radians = math.radians(angle_offset)
            spike_start_x = center_x + int((CELL_SIZE // 3) * math.cos(angle_radians))
            spike_start_y = center_y + int((CELL_SIZE // 3) * math.sin(angle_radians))
            spike_end_x = center_x + int((CELL_SIZE // 3 + spike_length) * math.cos(angle_radians))
            spike_end_y = center_y + int((CELL_SIZE // 3 + spike_length) * math.sin(angle_radians))
            pygame.draw.line(self.screen, COLOR_MONSTER,
                           (spike_start_x, spike_start_y),
                           (spike_end_x, spike_end_y), 3)
        
        # Eyes
        pygame.draw.circle(self.screen, (255, 255, 255),
                          (center_x - 6, center_y - 3), 5)
        pygame.draw.circle(self.screen, (255, 255, 255),
                          (center_x + 6, center_y - 3), 5)
        pygame.draw.circle(self.screen, (0, 0, 0),
                          (center_x - 6, center_y - 3), 2)
        pygame.draw.circle(self.screen, (0, 0, 0),
                          (center_x + 6, center_y - 3), 2)
    
    def close(self):
        """Clean up Pygame resources."""
        if self.render_enabled:
            pygame.quit()
    
    def get_action_space_size(self):
        """Return number of available actions."""
        return len(ACTIONS)
    
    def get_valid_actions(self):
        """Return list of valid actions from current position."""
        return list(ACTIONS)

