"""
Main entry point for Q-Learning and SARSA gridworld training.
Provides training, demonstration, and visualization capabilities.
"""

import sys
import time
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for compatibility
import matplotlib.pyplot as plt
from config import (
    TRAINING_EPISODES, ANIMATION_DELAY_TRAINING, ANIMATION_DELAY_DEMO
)
from gridworld import GridWorld
from q_learning import QLearningAgent
from sarsa import SarsaAgent


def train_agent(agent, environment, total_episodes, render_interval=0, verbose=True):
    """
    Train an agent on the environment.
    
    Args:
        agent: QLearningAgent or SarsaAgent
        environment: GridWorld environment
        total_episodes: Number of training episodes
        render_interval: Render every N episodes (0 for no rendering)
        verbose: Print progress updates
        
    Returns:
        Training statistics dictionary
    """
    start_time = time.time()
    successful_episodes = 0
    
    for episode_index in range(total_episodes):
        should_render = render_interval > 0 and episode_index % render_interval == 0
        
        reward, length, success = agent.train_episode(
            environment, 
            episode_index, 
            render=should_render,
            render_delay=ANIMATION_DELAY_TRAINING
        )
        
        if success:
            successful_episodes += 1
        
        # Progress update every 100 episodes
        if verbose and (episode_index + 1) % 100 == 0:
            recent_rewards = agent.episode_rewards[-100:]
            average_reward = sum(recent_rewards) / len(recent_rewards)
            print(f"Episode {episode_index + 1}/{total_episodes} | "
                  f"Avg Reward (last 100): {average_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Successful episodes: {successful_episodes}/{total_episodes} "
              f"({100 * successful_episodes / total_episodes:.1f}%)")
    
    return agent.get_statistics()


def plot_training_curves(statistics_q_learning, statistics_sarsa, level_index, save_path=None):
    """
    Plot training curves comparing Q-Learning and SARSA.
    
    Args:
        statistics_q_learning: Q-Learning training statistics
        statistics_sarsa: SARSA training statistics
        level_index: Level number for title
        save_path: Path to save plot (None for display only)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Smoothing window
    window_size = 50
    
    # Plot rewards
    axes[0].set_title(f"Level {level_index} - Episode Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    
    if statistics_q_learning and statistics_q_learning.get("episodes", 0) > 0:
        rewards_q = statistics_q_learning["total_rewards"]
        smoothed_q = smooth_data(rewards_q, window_size)
        axes[0].plot(smoothed_q, label="Q-Learning", color="#61afef", alpha=0.8)
    
    if statistics_sarsa and statistics_sarsa.get("episodes", 0) > 0:
        rewards_sarsa = statistics_sarsa["total_rewards"]
        smoothed_sarsa = smooth_data(rewards_sarsa, window_size)
        axes[0].plot(smoothed_sarsa, label="SARSA", color="#c678dd", alpha=0.8)
    
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axes[1].set_title(f"Level {level_index} - Episode Lengths")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    
    if statistics_q_learning and statistics_q_learning.get("episodes", 0) > 0:
        lengths_q = statistics_q_learning["episode_lengths"]
        smoothed_lengths_q = smooth_data(lengths_q, window_size)
        axes[1].plot(smoothed_lengths_q, label="Q-Learning", color="#61afef", alpha=0.8)
    
    if statistics_sarsa and statistics_sarsa.get("episodes", 0) > 0:
        lengths_sarsa = statistics_sarsa["episode_lengths"]
        smoothed_lengths_sarsa = smooth_data(lengths_sarsa, window_size)
        axes[1].plot(smoothed_lengths_sarsa, label="SARSA", color="#c678dd", alpha=0.8)
    
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def smooth_data(data, window_size):
    """
    Smooth data using moving average.
    
    Args:
        data: List of values
        window_size: Moving average window size
        
    Returns:
        Smoothed data list
    """
    if len(data) < window_size:
        return data
    
    smoothed = []
    for index in range(len(data)):
        start_index = max(0, index - window_size + 1)
        window = data[start_index:index + 1]
        smoothed.append(sum(window) / len(window))
    
    return smoothed


def run_level(level_index, algorithm="both", episodes=TRAINING_EPISODES, 
              demonstrate=True, plot=True):
    """
    Run training and demonstration for a specific level.
    
    Args:
        level_index: Level number (0-5)
        algorithm: "q_learning", "sarsa", or "both"
        episodes: Number of training episodes
        demonstrate: Whether to run visual demonstration after training
        plot: Whether to show training curves
    """
    print(f"\n{'='*60}")
    print(f"Level {level_index}")
    print(f"{'='*60}")
    
    statistics_q_learning = None
    statistics_sarsa = None
    q_agent = None
    sarsa_agent = None
    
    # Train Q-Learning
    if algorithm in ["q_learning", "both"]:
        print("\n--- Q-Learning Training ---")
        environment = GridWorld(level_index=level_index, render_enabled=False)
        q_agent = QLearningAgent()
        statistics_q_learning = train_agent(q_agent, environment, episodes)
        environment.close()
    
    # Train SARSA
    if algorithm in ["sarsa", "both"]:
        print("\n--- SARSA Training ---")
        environment = GridWorld(level_index=level_index, render_enabled=False)
        sarsa_agent = SarsaAgent()
        statistics_sarsa = train_agent(sarsa_agent, environment, episodes)
        environment.close()
    
    # Plot training curves
    if plot and (statistics_q_learning or statistics_sarsa):
        plot_training_curves(statistics_q_learning, statistics_sarsa, level_index)
    
    # Demonstrate learned policies
    if demonstrate:
        if q_agent is not None:
            print("\n--- Q-Learning Demonstration ---")
            print("Close the Pygame window to continue...")
            demo_environment = GridWorld(level_index=level_index, render_enabled=True)
            reward, success = q_agent.demonstrate(demo_environment, ANIMATION_DELAY_DEMO)
            print(f"Demonstration result: Reward={reward:.2f}, Success={success}")
            demo_environment.close()
        
        if sarsa_agent is not None:
            print("\n--- SARSA Demonstration ---")
            print("Close the Pygame window to continue...")
            demo_environment = GridWorld(level_index=level_index, render_enabled=True)
            reward, success = sarsa_agent.demonstrate(demo_environment, ANIMATION_DELAY_DEMO)
            print(f"Demonstration result: Reward={reward:.2f}, Success={success}")
            demo_environment.close()
    
    return statistics_q_learning, statistics_sarsa


def run_all_levels(episodes=TRAINING_EPISODES, demonstrate=True, plot=True):
    """
    Run training and demonstration for all levels.
    
    Args:
        episodes: Number of training episodes per level
        demonstrate: Whether to run visual demonstrations
        plot: Whether to show training curves
        
    Returns:
        Dictionary containing statistics for all levels (0-6)
    """
    all_statistics = {}
    
    for level_index in range(7):  # Levels 0-6
        if level_index == 6:
            # Level 6 uses intrinsic reward comparison
            level_6_statistics = run_level_6_intrinsic_comparison(episodes, demonstrate, plot)
            all_statistics[level_index] = level_6_statistics
        else:
            statistics_q, statistics_sarsa = run_level(
                level_index, 
                algorithm="both",
                episodes=episodes,
                demonstrate=demonstrate,
                plot=plot
            )
            all_statistics[level_index] = {
                "q_learning": statistics_q,
                "sarsa": statistics_sarsa
            }
    
    return all_statistics


def run_level_6_intrinsic_comparison(episodes=TRAINING_EPISODES, demonstrate=True, plot=True):
    """
    Run Level 6 with intrinsic reward comparison.
    Compares learning with and without intrinsic exploration bonus.
    
    Args:
        episodes: Number of training episodes
        demonstrate: Whether to run visual demonstrations
        plot: Whether to show training curves
    """
    level_index = 6
    
    print(f"\n{'='*60}")
    print(f"Level {level_index} - Intrinsic Reward Comparison")
    print(f"{'='*60}")
    
    # Train Q-Learning WITHOUT intrinsic reward
    print("\n--- Q-Learning (No Intrinsic Reward) ---")
    environment = GridWorld(level_index=level_index, render_enabled=False)
    q_agent_no_intrinsic = QLearningAgent(use_intrinsic_reward=False)
    statistics_q_no_intrinsic = train_agent(q_agent_no_intrinsic, environment, episodes)
    environment.close()
    
    # Train Q-Learning WITH intrinsic reward
    print("\n--- Q-Learning (With Intrinsic Reward) ---")
    environment = GridWorld(level_index=level_index, render_enabled=False)
    q_agent_with_intrinsic = QLearningAgent(use_intrinsic_reward=True)
    statistics_q_with_intrinsic = train_agent(q_agent_with_intrinsic, environment, episodes)
    environment.close()
    
    # Train SARSA WITHOUT intrinsic reward
    print("\n--- SARSA (No Intrinsic Reward) ---")
    environment = GridWorld(level_index=level_index, render_enabled=False)
    sarsa_agent_no_intrinsic = SarsaAgent(use_intrinsic_reward=False)
    statistics_sarsa_no_intrinsic = train_agent(sarsa_agent_no_intrinsic, environment, episodes)
    environment.close()
    
    # Train SARSA WITH intrinsic reward
    print("\n--- SARSA (With Intrinsic Reward) ---")
    environment = GridWorld(level_index=level_index, render_enabled=False)
    sarsa_agent_with_intrinsic = SarsaAgent(use_intrinsic_reward=True)
    statistics_sarsa_with_intrinsic = train_agent(sarsa_agent_with_intrinsic, environment, episodes)
    environment.close()
    
    # Plot comparison
    if plot:
        plot_intrinsic_comparison(
            statistics_q_no_intrinsic, statistics_q_with_intrinsic,
            statistics_sarsa_no_intrinsic, statistics_sarsa_with_intrinsic,
            level_index
        )
    
    # Demonstrate best policies
    if demonstrate:
        print("\n--- Q-Learning (With Intrinsic) Demonstration ---")
        print("Close the Pygame window to continue...")
        demo_environment = GridWorld(level_index=level_index, render_enabled=True)
        reward, success = q_agent_with_intrinsic.demonstrate(demo_environment, ANIMATION_DELAY_DEMO)
        print(f"Demonstration result: Reward={reward:.2f}, Success={success}")
        demo_environment.close()
        
        print("\n--- SARSA (With Intrinsic) Demonstration ---")
        print("Close the Pygame window to continue...")
        demo_environment = GridWorld(level_index=level_index, render_enabled=True)
        reward, success = sarsa_agent_with_intrinsic.demonstrate(demo_environment, ANIMATION_DELAY_DEMO)
        print(f"Demonstration result: Reward={reward:.2f}, Success={success}")
        demo_environment.close()
    
    return {
        "q_no_intrinsic": statistics_q_no_intrinsic,
        "q_with_intrinsic": statistics_q_with_intrinsic,
        "sarsa_no_intrinsic": statistics_sarsa_no_intrinsic,
        "sarsa_with_intrinsic": statistics_sarsa_with_intrinsic,
    }


def plot_intrinsic_comparison(statistics_q_no, statistics_q_with,
                               statistics_sarsa_no, statistics_sarsa_with,
                               level_index, save_path=None):
    """
    Plot training curves comparing with and without intrinsic reward.
    
    Args:
        statistics_q_no: Q-Learning without intrinsic reward
        statistics_q_with: Q-Learning with intrinsic reward
        statistics_sarsa_no: SARSA without intrinsic reward
        statistics_sarsa_with: SARSA with intrinsic reward
        level_index: Level number for title
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    window_size = 50
    
    # Q-Learning Rewards Comparison
    axes[0, 0].set_title(f"Level {level_index} - Q-Learning Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    
    if statistics_q_no and statistics_q_no.get("episodes", 0) > 0:
        smoothed = smooth_data(statistics_q_no["total_rewards"], window_size)
        axes[0, 0].plot(smoothed, label="Without Intrinsic", color="#e06c75", alpha=0.8)
    
    if statistics_q_with and statistics_q_with.get("episodes", 0) > 0:
        smoothed = smooth_data(statistics_q_with["total_rewards"], window_size)
        axes[0, 0].plot(smoothed, label="With Intrinsic", color="#61afef", alpha=0.8)
    
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Learning Episode Lengths Comparison
    axes[0, 1].set_title(f"Level {level_index} - Q-Learning Episode Lengths")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    
    if statistics_q_no and statistics_q_no.get("episodes", 0) > 0:
        smoothed = smooth_data(statistics_q_no["episode_lengths"], window_size)
        axes[0, 1].plot(smoothed, label="Without Intrinsic", color="#e06c75", alpha=0.8)
    
    if statistics_q_with and statistics_q_with.get("episodes", 0) > 0:
        smoothed = smooth_data(statistics_q_with["episode_lengths"], window_size)
        axes[0, 1].plot(smoothed, label="With Intrinsic", color="#61afef", alpha=0.8)
    
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SARSA Rewards Comparison
    axes[1, 0].set_title(f"Level {level_index} - SARSA Rewards")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Total Reward")
    
    if statistics_sarsa_no and statistics_sarsa_no.get("episodes", 0) > 0:
        smoothed = smooth_data(statistics_sarsa_no["total_rewards"], window_size)
        axes[1, 0].plot(smoothed, label="Without Intrinsic", color="#e5c07b", alpha=0.8)
    
    if statistics_sarsa_with and statistics_sarsa_with.get("episodes", 0) > 0:
        smoothed = smooth_data(statistics_sarsa_with["total_rewards"], window_size)
        axes[1, 0].plot(smoothed, label="With Intrinsic", color="#c678dd", alpha=0.8)
    
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # SARSA Episode Lengths Comparison
    axes[1, 1].set_title(f"Level {level_index} - SARSA Episode Lengths")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Steps")
    
    if statistics_sarsa_no and statistics_sarsa_no.get("episodes", 0) > 0:
        smoothed = smooth_data(statistics_sarsa_no["episode_lengths"], window_size)
        axes[1, 1].plot(smoothed, label="Without Intrinsic", color="#e5c07b", alpha=0.8)
    
    if statistics_sarsa_with and statistics_sarsa_with.get("episodes", 0) > 0:
        smoothed = smooth_data(statistics_sarsa_with["episode_lengths"], window_size)
        axes[1, 1].plot(smoothed, label="With Intrinsic", color="#c678dd", alpha=0.8)
    
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def interactive_mode():
    """
    Run interactive demonstration mode.
    Allows manual control of the agent using arrow keys.
    """
    import pygame
    
    print("\n--- Interactive Mode ---")
    print("Use arrow keys to move the agent.")
    print("Press R to reset, Q to quit, 0-6 to change levels.")
    
    current_level = 0
    environment = GridWorld(level_index=current_level, render_enabled=True)
    environment.reset()
    
    running = True
    while running:
        environment.render(algorithm_name="Interactive")
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = None
                
                if event.key == pygame.K_UP:
                    action = 0  # UP
                elif event.key == pygame.K_DOWN:
                    action = 1  # DOWN
                elif event.key == pygame.K_LEFT:
                    action = 2  # LEFT
                elif event.key == pygame.K_RIGHT:
                    action = 3  # RIGHT
                elif event.key == pygame.K_r:
                    environment.reset()
                    print("Environment reset")
                elif event.key == pygame.K_q:
                    running = False
                elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, 
                                  pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6]:
                    new_level = event.key - pygame.K_0
                    if new_level != current_level:
                        environment.close()
                        current_level = new_level
                        environment = GridWorld(level_index=current_level, render_enabled=True)
                        environment.reset()
                        print(f"Switched to Level {current_level}")
                
                if action is not None and not environment.is_done:
                    state, reward, done, info = environment.step(action)
                    if done:
                        if environment.is_dead:
                            print("Agent died! Press R to reset.")
                        else:
                            print("Level completed! Press R to reset.")
    
    environment.close()


def print_usage():
    """Print command line usage information."""
    print("""
Gridworld Q-Learning and SARSA Training

Usage:
    python main.py                     Run all levels with both algorithms
    python main.py --level N           Run specific level (0-6)
    python main.py --algorithm ALG     Use specific algorithm (q_learning, sarsa, both)
    python main.py --episodes N        Set number of training episodes
    python main.py --no-demo           Skip visual demonstrations
    python main.py --no-plot           Skip training curve plots
    python main.py --interactive       Run interactive mode with manual control
    python main.py --fast              Fast mode: no demos or plots
    python main.py --intrinsic         Run Level 6 intrinsic reward comparison
    
Examples:
    python main.py --level 0 --algorithm q_learning
    python main.py --level 4 --episodes 2000
    python main.py --fast
    python main.py --interactive
    python main.py --intrinsic --episodes 1500
""")


def main():
    """Main entry point with command line argument handling."""
    # Default settings
    level_index = None  # None means all levels
    algorithm = "both"
    episodes = TRAINING_EPISODES
    demonstrate = True
    plot = True
    interactive = False
    run_intrinsic_comparison = False
    
    # Parse command line arguments
    arguments = sys.argv[1:]
    index = 0
    
    while index < len(arguments):
        argument = arguments[index]
        
        if argument == "--help" or argument == "-h":
            print_usage()
            return
        elif argument == "--level":
            index += 1
            level_index = int(arguments[index])
        elif argument == "--algorithm":
            index += 1
            algorithm = arguments[index]
        elif argument == "--episodes":
            index += 1
            episodes = int(arguments[index])
        elif argument == "--no-demo":
            demonstrate = False
        elif argument == "--no-plot":
            plot = False
        elif argument == "--interactive":
            interactive = True
        elif argument == "--fast":
            demonstrate = False
            plot = False
        elif argument == "--intrinsic":
            run_intrinsic_comparison = True
        else:
            print(f"Unknown argument: {argument}")
            print_usage()
            return
        
        index += 1
    
    # Run appropriate mode
    if interactive:
        interactive_mode()
    elif run_intrinsic_comparison:
        run_level_6_intrinsic_comparison(episodes, demonstrate, plot)
    elif level_index is not None:
        if level_index == 6:
            run_level_6_intrinsic_comparison(episodes, demonstrate, plot)
        else:
            run_level(level_index, algorithm, episodes, demonstrate, plot)
    else:
        run_all_levels(episodes, demonstrate, plot)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

