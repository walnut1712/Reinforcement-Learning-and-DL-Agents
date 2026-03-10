"""
Core arena game logic for the Deep RL environment.
Contains Player, Enemy, Spawner, Projectile, and Arena classes.
"""

import math
import random
import pygame
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    PLAYER_RADIUS, PLAYER_MAX_HEALTH, PLAYER_SPEED, PLAYER_ROTATION_SPEED,
    PLAYER_FIRE_COOLDOWN, PLAYER_INITIAL_POSITION,
    ENEMY_RADIUS, ENEMY_HEALTH, ENEMY_SPEED, ENEMY_DAMAGE, ENEMY_COLLISION_DAMAGE,
    SPAWNER_RADIUS, SPAWNER_HEALTH, SPAWNER_SPAWN_INTERVAL, SPAWNER_MAX_ENEMIES,
    PROJECTILE_RADIUS, PROJECTILE_SPEED, PROJECTILE_DAMAGE, PROJECTILE_LIFETIME,
    INITIAL_SPAWNER_COUNT, SPAWNERS_PER_PHASE_INCREMENT, MAX_PHASE,
    REWARD_ENEMY_KILL, REWARD_SPAWNER_KILL, REWARD_PHASE_COMPLETE,
    REWARD_DAMAGE_TAKEN, REWARD_DEATH, REWARD_TIME_PENALTY,
    COLOR_BACKGROUND, COLOR_PLAYER, COLOR_PLAYER_DAMAGED, COLOR_ENEMY,
    COLOR_SPAWNER, COLOR_SPAWNER_ACTIVE, COLOR_PROJECTILE_PLAYER,
    COLOR_HEALTH_BAR, COLOR_HEALTH_BAR_BG, COLOR_TEXT
)


class Player:
    """
    Player ship controlled by the RL agent.
    Has position, velocity, orientation, health, and shooting capability.
    """
    
    def __init__(self):
        """Initialize player at center of arena."""
        self.position_x = PLAYER_INITIAL_POSITION[0]
        self.position_y = PLAYER_INITIAL_POSITION[1]
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.orientation = -90.0  # facing up initially (degrees)
        self.health = PLAYER_MAX_HEALTH
        self.fire_cooldown = 0
        self.is_alive = True
        self.damage_flash_timer = 0
    
    def reset(self):
        """Reset player to initial state."""
        self.position_x = PLAYER_INITIAL_POSITION[0]
        self.position_y = PLAYER_INITIAL_POSITION[1]
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.orientation = -90.0
        self.health = PLAYER_MAX_HEALTH
        self.fire_cooldown = 0
        self.is_alive = True
        self.damage_flash_timer = 0
    
    def apply_thrust(self):
        """Apply forward thrust in current orientation direction."""
        radians = math.radians(self.orientation)
        self.velocity_x = math.cos(radians) * PLAYER_SPEED
        self.velocity_y = math.sin(radians) * PLAYER_SPEED
    
    def rotate_left(self):
        """Rotate player counter-clockwise."""
        self.orientation -= PLAYER_ROTATION_SPEED
    
    def rotate_right(self):
        """Rotate player clockwise."""
        self.orientation += PLAYER_ROTATION_SPEED
    
    def move_direction(self, direction_x, direction_y):
        """Move player in specified direction (for directional control)."""
        self.velocity_x = direction_x * PLAYER_SPEED
        self.velocity_y = direction_y * PLAYER_SPEED
        # Update orientation to face movement direction
        if direction_x != 0 or direction_y != 0:
            self.orientation = math.degrees(math.atan2(direction_y, direction_x))
    
    def stop(self):
        """Stop player movement."""
        self.velocity_x = 0.0
        self.velocity_y = 0.0
    
    def update(self):
        """Update player position and state."""
        # Apply velocity
        self.position_x += self.velocity_x
        self.position_y += self.velocity_y
        
        # Keep player within arena bounds
        self.position_x = max(PLAYER_RADIUS, min(WINDOW_WIDTH - PLAYER_RADIUS, self.position_x))
        self.position_y = max(PLAYER_RADIUS, min(WINDOW_HEIGHT - PLAYER_RADIUS, self.position_y))
        
        # Update cooldowns
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        
        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1
        
        # Apply friction to slow down
        self.velocity_x *= 0.9
        self.velocity_y *= 0.9
    
    def can_shoot(self):
        """Check if player can fire a projectile."""
        return self.fire_cooldown == 0 and self.is_alive
    
    def shoot(self):
        """Create a projectile in current orientation direction."""
        if not self.can_shoot():
            return None
        
        self.fire_cooldown = PLAYER_FIRE_COOLDOWN
        radians = math.radians(self.orientation)
        
        return Projectile(
            self.position_x + math.cos(radians) * PLAYER_RADIUS,
            self.position_y + math.sin(radians) * PLAYER_RADIUS,
            math.cos(radians) * PROJECTILE_SPEED,
            math.sin(radians) * PROJECTILE_SPEED,
            is_player_projectile=True
        )
    
    def take_damage(self, damage_amount):
        """Apply damage to player."""
        self.health -= damage_amount
        self.damage_flash_timer = 10
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
    
    def draw(self, screen):
        """Render player on screen."""
        # Choose color based on damage state
        color = COLOR_PLAYER_DAMAGED if self.damage_flash_timer > 0 else COLOR_PLAYER
        
        # Draw player as triangle pointing in orientation direction
        radians = math.radians(self.orientation)
        
        # Triangle vertices
        front_x = self.position_x + math.cos(radians) * PLAYER_RADIUS
        front_y = self.position_y + math.sin(radians) * PLAYER_RADIUS
        
        back_left_angle = radians + math.radians(140)
        back_left_x = self.position_x + math.cos(back_left_angle) * PLAYER_RADIUS
        back_left_y = self.position_y + math.sin(back_left_angle) * PLAYER_RADIUS
        
        back_right_angle = radians - math.radians(140)
        back_right_x = self.position_x + math.cos(back_right_angle) * PLAYER_RADIUS
        back_right_y = self.position_y + math.sin(back_right_angle) * PLAYER_RADIUS
        
        pygame.draw.polygon(screen, color, [
            (front_x, front_y),
            (back_left_x, back_left_y),
            (back_right_x, back_right_y)
        ])
        
        # Draw health bar above player
        health_bar_width = 30
        health_bar_height = 4
        health_ratio = self.health / PLAYER_MAX_HEALTH
        
        bar_x = self.position_x - health_bar_width // 2
        bar_y = self.position_y - PLAYER_RADIUS - 10
        
        pygame.draw.rect(screen, COLOR_HEALTH_BAR_BG, 
                        (bar_x, bar_y, health_bar_width, health_bar_height))
        pygame.draw.rect(screen, COLOR_HEALTH_BAR, 
                        (bar_x, bar_y, int(health_bar_width * health_ratio), health_bar_height))


class Enemy:
    """
    Enemy entity that navigates toward the player.
    Spawned by Spawner objects.
    """
    
    def __init__(self, position_x, position_y):
        """Initialize enemy at specified position."""
        self.position_x = position_x
        self.position_y = position_y
        self.health = ENEMY_HEALTH
        self.is_alive = True
        self.spawner_id = None  # track which spawner created this enemy
    
    def update(self, player_x, player_y):
        """Move toward player position."""
        if not self.is_alive:
            return
        
        # Calculate direction to player
        delta_x = player_x - self.position_x
        delta_y = player_y - self.position_y
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
        
        if distance > 0:
            # Normalize and apply speed
            self.position_x += (delta_x / distance) * ENEMY_SPEED
            self.position_y += (delta_y / distance) * ENEMY_SPEED
    
    def take_damage(self, damage_amount):
        """Apply damage to enemy."""
        self.health -= damage_amount
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
    
    def check_collision_with_player(self, player):
        """Check if enemy collides with player."""
        distance = math.sqrt(
            (self.position_x - player.position_x) ** 2 +
            (self.position_y - player.position_y) ** 2
        )
        return distance < (ENEMY_RADIUS + PLAYER_RADIUS)
    
    def draw(self, screen):
        """Render enemy on screen."""
        if not self.is_alive:
            return
        
        pygame.draw.circle(screen, COLOR_ENEMY, 
                          (int(self.position_x), int(self.position_y)), ENEMY_RADIUS)
        
        # Draw health bar
        health_bar_width = 20
        health_bar_height = 3
        health_ratio = self.health / ENEMY_HEALTH
        
        bar_x = self.position_x - health_bar_width // 2
        bar_y = self.position_y - ENEMY_RADIUS - 8
        
        pygame.draw.rect(screen, COLOR_HEALTH_BAR_BG, 
                        (bar_x, bar_y, health_bar_width, health_bar_height))
        pygame.draw.rect(screen, COLOR_HEALTH_BAR, 
                        (bar_x, bar_y, int(health_bar_width * health_ratio), health_bar_height))


class Spawner:
    """
    Stationary spawner that periodically creates enemies.
    Destroying all spawners advances to next phase.
    """
    
    def __init__(self, position_x, position_y, spawner_id):
        """Initialize spawner at specified position."""
        self.position_x = position_x
        self.position_y = position_y
        self.spawner_id = spawner_id
        self.health = SPAWNER_HEALTH
        self.spawn_timer = SPAWNER_SPAWN_INTERVAL
        self.is_alive = True
        self.active_enemy_count = 0
        self.pulse_timer = 0  # for visual effect
    
    def update(self):
        """Update spawn timer and visual effects."""
        if not self.is_alive:
            return None
        
        self.pulse_timer += 1
        self.spawn_timer -= 1
        
        if self.spawn_timer <= 0 and self.active_enemy_count < SPAWNER_MAX_ENEMIES:
            self.spawn_timer = SPAWNER_SPAWN_INTERVAL
            return self.spawn_enemy()
        
        return None
    
    def spawn_enemy(self):
        """Create a new enemy near the spawner."""
        # Spawn at random position around spawner
        angle = random.uniform(0, 2 * math.pi)
        spawn_distance = SPAWNER_RADIUS + ENEMY_RADIUS + 10
        
        enemy_x = self.position_x + math.cos(angle) * spawn_distance
        enemy_y = self.position_y + math.sin(angle) * spawn_distance
        
        # Keep within bounds
        enemy_x = max(ENEMY_RADIUS, min(WINDOW_WIDTH - ENEMY_RADIUS, enemy_x))
        enemy_y = max(ENEMY_RADIUS, min(WINDOW_HEIGHT - ENEMY_RADIUS, enemy_y))
        
        enemy = Enemy(enemy_x, enemy_y)
        enemy.spawner_id = self.spawner_id
        self.active_enemy_count += 1
        
        return enemy
    
    def on_enemy_destroyed(self):
        """Called when an enemy from this spawner is destroyed."""
        self.active_enemy_count = max(0, self.active_enemy_count - 1)
    
    def take_damage(self, damage_amount):
        """Apply damage to spawner."""
        self.health -= damage_amount
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
    
    def check_collision_with_player(self, player):
        """Check if spawner collides with player."""
        dx = self.position_x - player.position_x
        dy = self.position_y - player.position_y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance < (SPAWNER_RADIUS + PLAYER_RADIUS)
    
    def draw(self, screen):
        """Render spawner on screen."""
        if not self.is_alive:
            return
        
        # Pulsing effect when about to spawn
        pulse = abs(math.sin(self.pulse_timer * 0.1)) * 5
        color = COLOR_SPAWNER_ACTIVE if self.spawn_timer < 30 else COLOR_SPAWNER
        
        # Draw spawner as hexagon
        points = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            point_x = self.position_x + math.cos(angle) * (SPAWNER_RADIUS + pulse)
            point_y = self.position_y + math.sin(angle) * (SPAWNER_RADIUS + pulse)
            points.append((point_x, point_y))
        
        pygame.draw.polygon(screen, color, points)
        pygame.draw.polygon(screen, (255, 255, 255), points, 2)
        
        # Draw health bar
        health_bar_width = 40
        health_bar_height = 5
        health_ratio = self.health / SPAWNER_HEALTH
        
        bar_x = self.position_x - health_bar_width // 2
        bar_y = self.position_y - SPAWNER_RADIUS - 12
        
        pygame.draw.rect(screen, COLOR_HEALTH_BAR_BG, 
                        (bar_x, bar_y, health_bar_width, health_bar_height))
        pygame.draw.rect(screen, COLOR_HEALTH_BAR, 
                        (bar_x, bar_y, int(health_bar_width * health_ratio), health_bar_height))


class Projectile:
    """
    Projectile fired by player.
    Travels in straight line and damages enemies/spawners on contact.
    """
    
    def __init__(self, position_x, position_y, velocity_x, velocity_y, is_player_projectile=True):
        """Initialize projectile at position with velocity."""
        self.position_x = position_x
        self.position_y = position_y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.is_player_projectile = is_player_projectile
        self.lifetime = PROJECTILE_LIFETIME
        self.is_active = True
    
    def update(self):
        """Update projectile position and lifetime."""
        self.position_x += self.velocity_x
        self.position_y += self.velocity_y
        self.lifetime -= 1
        
        # Deactivate if out of bounds or expired
        if (self.position_x < 0 or self.position_x > WINDOW_WIDTH or
            self.position_y < 0 or self.position_y > WINDOW_HEIGHT or
            self.lifetime <= 0):
            self.is_active = False
    
    def check_collision_with_enemy(self, enemy):
        """Check collision with enemy."""
        distance = math.sqrt(
            (self.position_x - enemy.position_x) ** 2 +
            (self.position_y - enemy.position_y) ** 2
        )
        return distance < (PROJECTILE_RADIUS + ENEMY_RADIUS)
    
    def check_collision_with_spawner(self, spawner):
        """Check collision with spawner."""
        distance = math.sqrt(
            (self.position_x - spawner.position_x) ** 2 +
            (self.position_y - spawner.position_y) ** 2
        )
        return distance < (PROJECTILE_RADIUS + SPAWNER_RADIUS)
    
    def draw(self, screen):
        """Render projectile on screen."""
        if not self.is_active:
            return
        
        color = COLOR_PROJECTILE_PLAYER if self.is_player_projectile else COLOR_PROJECTILE_PLAYER
        pygame.draw.circle(screen, color, 
                          (int(self.position_x), int(self.position_y)), PROJECTILE_RADIUS)


class Arena:
    """
    Main game state manager.
    Handles all entities, collisions, phases, and game logic.
    """
    
    def __init__(self):
        """Initialize arena with empty state."""
        self.player = Player()
        self.enemies = []
        self.spawners = []
        self.projectiles = []
        self.current_phase = 1
        self.step_count = 0
        self.total_enemies_killed = 0
        self.total_spawners_killed = 0
        self.is_game_over = False
        self.spawner_id_counter = 0
    
    def reset(self):
        """Reset arena to initial state for new episode."""
        self.player.reset()
        self.enemies = []
        self.spawners = []
        self.projectiles = []
        self.current_phase = 1
        self.step_count = 0
        self.total_enemies_killed = 0
        self.total_spawners_killed = 0
        self.is_game_over = False
        self.spawner_id_counter = 0
        
        self._spawn_phase_spawners()
    
    def _spawn_phase_spawners(self):
        """Spawn spawners for current phase."""
        spawner_count = INITIAL_SPAWNER_COUNT + (self.current_phase - 1) * SPAWNERS_PER_PHASE_INCREMENT
        spawner_count = min(spawner_count, 6)  # cap at 6 spawners
        
        for i in range(spawner_count):
            # Place spawners around arena edges, avoiding player start position
            attempts = 0
            while attempts < 50:
                margin = SPAWNER_RADIUS + 50
                position_x = random.uniform(margin, WINDOW_WIDTH - margin)
                position_y = random.uniform(margin, WINDOW_HEIGHT - margin)
                
                # Ensure distance from player start
                distance_from_center = math.sqrt(
                    (position_x - PLAYER_INITIAL_POSITION[0]) ** 2 +
                    (position_y - PLAYER_INITIAL_POSITION[1]) ** 2
                )
                
                # Ensure distance from other spawners
                too_close = False
                for spawner in self.spawners:
                    distance = math.sqrt(
                        (position_x - spawner.position_x) ** 2 +
                        (position_y - spawner.position_y) ** 2
                    )
                    if distance < SPAWNER_RADIUS * 4:
                        too_close = True
                        break
                
                if distance_from_center > 150 and not too_close:
                    spawner = Spawner(position_x, position_y, self.spawner_id_counter)
                    self.spawner_id_counter += 1
                    self.spawners.append(spawner)
                    break
                
                attempts += 1
    
    def step(self, player_action_callback):
        """
        Execute one game step.
        player_action_callback: function that takes player and returns action result
        Returns: dictionary with step results and rewards
        """
        step_reward = 0.0
        phase_completed = False
        hits_dealt = 0
        spawner_hits = 0
        
        self.step_count += 1
        
        # Apply player action
        player_action_callback(self.player)
        
        # Update player
        self.player.update()
        
        # Update spawners and spawn enemies
        for spawner in self.spawners:
            if spawner.is_alive:
                new_enemy = spawner.update()
                if new_enemy is not None:
                    self.enemies.append(new_enemy)
                
                # Check collision with player
                if spawner.check_collision_with_player(self.player):
                    self.player.take_damage(ENEMY_COLLISION_DAMAGE)
                    step_reward += REWARD_DAMAGE_TAKEN
        
        # Update enemies
        for enemy in self.enemies:
            if enemy.is_alive:
                enemy.update(self.player.position_x, self.player.position_y)
                
                # Check collision with player
                if enemy.check_collision_with_player(self.player):
                    self.player.take_damage(ENEMY_COLLISION_DAMAGE)
                    enemy.is_alive = False
                    step_reward += REWARD_DAMAGE_TAKEN
                    
                    # Update spawner enemy count
                    for spawner in self.spawners:
                        if spawner.spawner_id == enemy.spawner_id:
                            spawner.on_enemy_destroyed()
                            break
        
        # Update projectiles
        for projectile in self.projectiles:
            if projectile.is_active:
                projectile.update()
                
                if projectile.is_player_projectile:
                    # Check collision with enemies
                    for enemy in self.enemies:
                        if enemy.is_alive and projectile.check_collision_with_enemy(enemy):
                            enemy.take_damage(PROJECTILE_DAMAGE)
                            projectile.is_active = False
                            hits_dealt += 1
                            
                            if not enemy.is_alive:
                                self.total_enemies_killed += 1
                                step_reward += REWARD_ENEMY_KILL
                                
                                # Update spawner enemy count
                                for spawner in self.spawners:
                                    if spawner.spawner_id == enemy.spawner_id:
                                        spawner.on_enemy_destroyed()
                                        break
                            break
                    
                    # Check collision with spawners
                    for spawner in self.spawners:
                        if spawner.is_alive and projectile.check_collision_with_spawner(spawner):
                            spawner.take_damage(PROJECTILE_DAMAGE)
                            projectile.is_active = False
                            hits_dealt += 1
                            spawner_hits += 1
                            
                            if not spawner.is_alive:
                                self.total_spawners_killed += 1
                                step_reward += REWARD_SPAWNER_KILL
                            break
        
        # Clean up dead entities
        self.enemies = [e for e in self.enemies if e.is_alive]
        self.projectiles = [p for p in self.projectiles if p.is_active]
        
        # Check phase completion
        active_spawners = [s for s in self.spawners if s.is_alive]
        if len(active_spawners) == 0:
            if self.current_phase < MAX_PHASE:
                self.current_phase += 1
                step_reward += REWARD_PHASE_COMPLETE
                phase_completed = True
                self._spawn_phase_spawners()
            else:
                # Won the game
                self.is_game_over = True
                step_reward += REWARD_PHASE_COMPLETE * 2
        
        # Check player death
        if not self.player.is_alive:
            self.is_game_over = True
            step_reward += REWARD_DEATH
        
        # Time penalty
        step_reward += REWARD_TIME_PENALTY
        
        return {
            "reward": step_reward,
            "done": self.is_game_over,
            "phase_completed": phase_completed,
            "enemies_killed": self.total_enemies_killed,
            "spawners_killed": self.total_spawners_killed,
            "current_phase": self.current_phase,
            "hits_dealt": hits_dealt,
            "spawner_hits": spawner_hits
        }
    
    def add_projectile(self, projectile):
        """Add a projectile to the arena."""
        if projectile is not None:
            self.projectiles.append(projectile)
    
    def get_nearest_enemy(self):
        """Get nearest enemy to player. Returns (distance, direction_x, direction_y) or None."""
        if not self.enemies:
            return None
        
        nearest_distance = float('inf')
        nearest_direction_x = 0.0
        nearest_direction_y = 0.0
        
        for enemy in self.enemies:
            if not enemy.is_alive:
                continue
            
            delta_x = enemy.position_x - self.player.position_x
            delta_y = enemy.position_y - self.player.position_y
            distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
            
            if distance < nearest_distance:
                nearest_distance = distance
                if distance > 0:
                    nearest_direction_x = delta_x / distance
                    nearest_direction_y = delta_y / distance
        
        if nearest_distance == float('inf'):
            return None
        
        return (nearest_distance, nearest_direction_x, nearest_direction_y)
    
    def get_nearest_spawner(self):
        """Get nearest alive spawner to player. Returns (distance, direction_x, direction_y) or None."""
        alive_spawners = [s for s in self.spawners if s.is_alive]
        
        if not alive_spawners:
            return None
        
        nearest_distance = float('inf')
        nearest_direction_x = 0.0
        nearest_direction_y = 0.0
        
        for spawner in alive_spawners:
            delta_x = spawner.position_x - self.player.position_x
            delta_y = spawner.position_y - self.player.position_y
            distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
            
            if distance < nearest_distance:
                nearest_distance = distance
                if distance > 0:
                    nearest_direction_x = delta_x / distance
                    nearest_direction_y = delta_y / distance
        
        return (nearest_distance, nearest_direction_x, nearest_direction_y)
    
    def get_enemy_count(self):
        """Get count of alive enemies."""
        return len([e for e in self.enemies if e.is_alive])
    
    def get_spawner_count(self):
        """Get count of alive spawners."""
        return len([s for s in self.spawners if s.is_alive])
    
    def draw(self, screen):
        """Render all arena entities."""
        screen.fill(COLOR_BACKGROUND)
        
        # Draw spawners
        for spawner in self.spawners:
            spawner.draw(screen)
        
        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(screen)
        
        # Draw projectiles
        for projectile in self.projectiles:
            projectile.draw(screen)
        
        # Draw player
        self.player.draw(screen)
        
        # Draw HUD
        self._draw_hud(screen)
    
    def _draw_hud(self, screen):
        """Draw heads-up display with game info."""
        font = pygame.font.Font(None, 24)
        
        # Phase info
        phase_text = font.render(f"Phase: {self.current_phase}/{MAX_PHASE}", True, COLOR_TEXT)
        screen.blit(phase_text, (10, 10))
        
        # Enemy count
        enemy_text = font.render(f"Enemies: {self.get_enemy_count()}", True, COLOR_TEXT)
        screen.blit(enemy_text, (10, 35))
        
        # Spawner count
        spawner_text = font.render(f"Spawners: {self.get_spawner_count()}", True, COLOR_TEXT)
        screen.blit(spawner_text, (10, 60))
        
        # Player health
        health_text = font.render(f"Health: {self.player.health}/{PLAYER_MAX_HEALTH}", True, COLOR_TEXT)
        screen.blit(health_text, (WINDOW_WIDTH - 120, 10))
        
        # Kill counts
        kills_text = font.render(f"Kills: {self.total_enemies_killed} | Spawners: {self.total_spawners_killed}", 
                                True, COLOR_TEXT)
        screen.blit(kills_text, (WINDOW_WIDTH // 2 - 80, 10))
