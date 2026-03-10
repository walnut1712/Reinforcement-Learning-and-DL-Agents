"""
Evaluation script for trained agents.
Loads models and runs visual demonstrations in the arena.
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from rotation_env import RotationEnv
from directional_env import DirectionalEnv


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument(
        "--model", type=str, default="rotation",
        choices=["rotation", "directional"],
        help="Which model to evaluate"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to model file (overrides --model)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Use random actions instead of trained model"
    )
    return parser.parse_args()


def run_evaluation():
    """Run evaluation with visual rendering."""
    args = parse_arguments()
    
    # Select environment
    if args.model == "rotation":
        environment = RotationEnv(render_mode="human")
        default_model_path = "models/rotation_model.zip"
    else:
        environment = DirectionalEnv(render_mode="human")
        default_model_path = "models/directional_model.zip"
    
    # Load model if not using random
    model = None
    if not args.random:
        model_path = args.model_path if args.model_path else default_model_path
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Use --random flag to run with random actions, or train a model first.")
            environment.close()
            return
        
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=environment)
    
    print(f"\nRunning {args.episodes} episodes with {'random' if args.random else 'trained'} agent")
    print("Press Ctrl+C or close window to stop\n")
    
    total_rewards = []
    total_enemies_killed = []
    total_spawners_killed = []
    max_phases_reached = []
    
    try:
        for episode_index in range(args.episodes):
            observation, info = environment.reset()
            episode_reward = 0.0
            step_count = 0
            done = False
            
            while not done:
                # Render
                environment.render()
                
                # Select action
                if args.random:
                    action = environment.action_space.sample()
                else:
                    action, _ = model.predict(observation, deterministic=True)
                
                # Execute action
                observation, reward, terminated, truncated, info = environment.step(action)
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
            
            # Record statistics
            total_rewards.append(episode_reward)
            total_enemies_killed.append(info.get("enemies_killed", 0))
            total_spawners_killed.append(info.get("spawners_killed", 0))
            max_phases_reached.append(info.get("phase", 1))
            
            print(f"Episode {episode_index + 1}/{args.episodes}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Enemies killed: {info.get('enemies_killed', 0)}")
            print(f"  Spawners killed: {info.get('spawners_killed', 0)}")
            print(f"  Phase reached: {info.get('phase', 1)}")
            print()
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    
    finally:
        environment.close()
    
    # Print summary
    if total_rewards:
        print("\n=== Evaluation Summary ===")
        print(f"Episodes completed: {len(total_rewards)}")
        print(f"Average reward: {np.mean(total_rewards):.2f}")
        print(f"Average enemies killed: {np.mean(total_enemies_killed):.1f}")
        print(f"Average spawners killed: {np.mean(total_spawners_killed):.1f}")
        print(f"Average phase reached: {np.mean(max_phases_reached):.1f}")


if __name__ == "__main__":
    run_evaluation()
