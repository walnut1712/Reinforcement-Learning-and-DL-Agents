"""
Rotation-based control scheme environment.
Player uses thrust and rotation to move, like a spaceship.
"""

from gymnasium import spaces
from arena_env import ArenaEnv


class RotationEnv(ArenaEnv):
    """
    Arena environment with rotation-based controls.
    
    Action space: MultiDiscrete([2, 3, 2])
    - thrust: 0=none, 1=thrust
    - rotate: 0=none, 1=left, 2=right
    - shoot: 0=none, 1=shoot
    """
    
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.action_space = spaces.MultiDiscrete([2, 3, 2])
    
    def _apply_action(self, player, action):
        """Apply rotation-based action to player."""
        thrust_action = action[0]
        rotate_action = action[1]
        shoot_action = action[2]
        
        # 1. Thrust
        if thrust_action == 1:
            player.apply_thrust()
            
        # 2. Rotation
        if rotate_action == 1:
            player.rotate_left()
        elif rotate_action == 2:
            player.rotate_right()
        
        # 3. Shooting
        if shoot_action == 1:
            projectile = player.shoot()
            if projectile:
                self.arena.add_projectile(projectile)


# Register the environment for use with Stable Baselines3
def make_rotation_env(render_mode=None):
    """Factory function to create rotation environment."""
    return RotationEnv(render_mode=render_mode)
