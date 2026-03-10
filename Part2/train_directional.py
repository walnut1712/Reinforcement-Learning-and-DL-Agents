"""
Training script for directional control agent.
Uses PPO from Stable Baselines3 with TensorBoard logging.
"""

import os
import sys
import argparse
import subprocess
import threading
import webbrowser
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from directional_env import DirectionalEnv


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train directional control agent")
    parser.add_argument(
        "--timesteps", type=int, default=500000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.0003,
        help="Learning rate for PPO"
    )
    parser.add_argument(
        "--save-freq", type=int, default=50000,
        help="Save checkpoint every N timesteps"
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable automatic TensorBoard launch"
    )
    return parser.parse_args()


def launch_tensorboard(log_directory):
    """Launch TensorBoard in a background process."""
    import socket
    
    # Convert to absolute path
    abs_log_dir = os.path.abspath(log_directory)
    
    def is_port_open(port):
        """Check if a service is running on the port."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False
    
    try:
        # Check if TensorBoard is already running
        if is_port_open(6006):
            print("TensorBoard is already running on port 6006")
            webbrowser.open("http://localhost:6006")
            return None  # No new process to manage
        
        print(f"Starting TensorBoard with logdir: {abs_log_dir}")
        
        # Start TensorBoard as subprocess
        process = subprocess.Popen(
            f"tensorboard --logdir \"{abs_log_dir}\" --port 6006 --bind_all",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        
        # Wait for TensorBoard to be ready (up to 15 seconds)
        for attempt in range(15):
            time.sleep(1)
            if is_port_open(6006):
                print("TensorBoard is ready!")
                break
            # Check if process failed
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"TensorBoard failed to start:")
                print(f"  stderr: {stderr.decode()}")
                return None
            print(f"  Waiting for TensorBoard... ({attempt + 1}/15)")
        
        # Open browser
        webbrowser.open("http://localhost:6006")
        print("TensorBoard launched at http://localhost:6006")
        
        return process
    except Exception as error:
        print(f"Could not launch TensorBoard: {error}")
        print("You can start it manually: tensorboard --logdir=logs")
        return None


def train():
    """Train the directional control agent."""
    args = parse_arguments()
    
    # Create directories
    models_directory = "models"
    logs_directory = "logs/directional"
    os.makedirs(models_directory, exist_ok=True)
    os.makedirs(logs_directory, exist_ok=True)
    
    # Create environment (no rendering during training)
    environment = DirectionalEnv(render_mode=None)
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        environment,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=128,  # Larger batch for stability
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Higher entropy for more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=logs_directory,
        policy_kwargs={
            "net_arch": [256, 256]  # Larger network for complex control
        }
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=models_directory,
        name_prefix="directional_checkpoint"
    )
    
    print(f"Starting training for {args.timesteps} timesteps...")
    print(f"TensorBoard logs: {logs_directory}")
    print(f"Model checkpoints: {models_directory}")
    
    # Launch TensorBoard automatically
    tensorboard_process = None
    if not args.no_tensorboard:
        tensorboard_process = launch_tensorboard(logs_directory)
    
    # Train the agent
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(models_directory, "directional_model")
    model.save(final_model_path)
    print(f"Training complete. Model saved to {final_model_path}.zip")
    
    # Cleanup
    environment.close()
    
    # Terminate TensorBoard
    if tensorboard_process is not None:
        print("Shutting down TensorBoard...")
        try:
            import platform
            if platform.system() == "Windows":
                # On Windows, kill the process tree
                subprocess.run(
                    f"taskkill /F /T /PID {tensorboard_process.pid}",
                    shell=True,
                    capture_output=True
                )
            else:
                # On Linux/Mac, terminate process group
                import signal
                os.killpg(os.getpgid(tensorboard_process.pid), signal.SIGTERM)
            tensorboard_process.wait(timeout=5)
        except Exception as e:
            print(f"Note: Could not fully terminate TensorBoard: {e}")


if __name__ == "__main__":
    train()
