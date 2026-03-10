"""
Directional control scheme environment.
Player uses WASD-style movement in four directions.
"""

from gymnasium import spaces
from arena_env import ArenaEnv


class DirectionalEnv(ArenaEnv):
    """
    Arena environment with directional controls.
    
    Action space: MultiDiscrete([5, 2])
    - movement: 0=stop, 1=up, 2=down, 3=left, 4=right
    - shoot: 0=none, 1=shoot
    """
    
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.action_space = spaces.MultiDiscrete([5, 2])
    
    def _apply_action(self, player, action):
        """Apply directional action to player."""
        movement_action = action[0]
        shoot_action = action[1]
        
        # Apply movement
        if movement_action == 0:
            # No movement - stop
            player.stop()
        elif movement_action == 1:
            # Move up
            player.move_direction(0, -1)
        elif movement_action == 2:
            # Move down
            player.move_direction(0, 1)
        elif movement_action == 3:
            # Move left
            player.move_direction(-1, 0)
        elif movement_action == 4:
            # Move right
            player.move_direction(1, 0)
        
        # Apply shooting (can happen simultaneously with movement)
        if shoot_action == 1:
            projectile = player.shoot()
            self.arena.add_projectile(projectile)


# Register the environment for use with Stable Baselines3
def make_directional_env(render_mode=None):
    """Factory function to create directional environment."""
    return DirectionalEnv(render_mode=render_mode)
