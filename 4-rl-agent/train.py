"""
Main training script for RL agents

This script demonstrates how to train different RL agents on various environments.
"""

import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime
import os
import json

from grid_world import GridWorldEnv, SimpleFrozenLake
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent


def train_q_learning_grid_world():
    """Train Q-Learning agent on Grid World environment."""
    print("=== Training Q-Learning Agent on Grid World ===\n")
    
    # Create environment
    env = GridWorldEnv(grid_size=10, render_mode=None)
    
    # Create agent
    n_states = env.grid_size * env.grid_size
    n_actions = env.action_space.n
    
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42
    )
    
    # Train
    print("Training...")
    history = agent.train(
        env,
        n_episodes=2000,
        max_steps_per_episode=200,
        verbose=True,
        render_frequency=0
    )
    
    # Evaluate
    print("\nEvaluating...")
    eval_results = agent.evaluate(
        env,
        n_episodes=100,
        render=False
    )
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {eval_results['mean_length']:.1f}")
    print(f"  Success Rate: {eval_results['success_rate']:.2%}")
    
    # Save results
    save_dir = "results/q_learning_grid_world"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save agent
    agent.save(f"{save_dir}/agent.pkl")
    
    # Save training history
    with open(f"{save_dir}/history.json", 'w') as f:
        json.dump({
            'episode_rewards': history['episode_rewards'],
            'episode_lengths': history['episode_lengths'],
            'eval_results': eval_results
        }, f, indent=2)
    
    # Plot results
    agent.plot_training_history()
    
    # Visualize learned policy
    print("\nLearned Policy:")
    env_vis = GridWorldEnv(grid_size=10, render_mode='human')
    agent.visualize_policy(env_vis, grid_size=10)
    
    return agent, env


def train_dqn_grid_world():
    """Train DQN agent on Grid World environment."""
    print("=== Training DQN Agent on Grid World ===\n")
    
    # Create environment
    env = GridWorldEnv(grid_size=10, render_mode=None)
    
    # Create agent
    state_size = 2  # (x, y) coordinates
    action_size = env.action_space.n
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        batch_size=32,
        update_target_every=100,
        hidden_size=128,
        device='cpu',
        seed=42
    )
    
    # Train
    print("Training...")
    history = agent.train(
        env,
        n_episodes=1000,
        max_steps_per_episode=200,
        verbose=True,
        render_frequency=0
    )
    
    # Evaluate
    print("\nEvaluating...")
    eval_results = agent.evaluate(
        env,
        n_episodes=100,
        render=False
    )
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {eval_results['mean_length']:.1f}")
    
    # Save results
    save_dir = "results/dqn_grid_world"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save agent
    agent.save(f"{save_dir}/model.pth")
    
    # Save training history
    with open(f"{save_dir}/history.json", 'w') as f:
        json.dump({
            'episode_rewards': history['episode_rewards'],
            'episode_lengths': history['episode_lengths'],
            'losses': history['losses'],
            'eval_results': eval_results
        }, f, indent=2)
    
    # Plot results
    agent.plot_training_history()
    
    return agent, env


def train_q_learning_cartpole():
    """Train Q-Learning agent on CartPole (discretized)."""
    print("=== Training Q-Learning Agent on CartPole ===\n")
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Discretize the continuous state space
    n_bins = [20, 20, 20, 20]  # bins for each state dimension
    state_bounds = [
        [-2.4, 2.4],     # cart position
        [-3.0, 3.0],     # cart velocity
        [-0.21, 0.21],   # pole angle
        [-2.0, 2.0]      # pole angular velocity
    ]
    
    def discretize_state(state):
        """Convert continuous state to discrete."""
        discrete_state = []
        for i, s in enumerate(state):
            # Clip to bounds
            s = np.clip(s, state_bounds[i][0], state_bounds[i][1])
            # Discretize
            bin_width = (state_bounds[i][1] - state_bounds[i][0]) / n_bins[i]
            discrete_s = int((s - state_bounds[i][0]) / bin_width)
            discrete_s = min(discrete_s, n_bins[i] - 1)
            discrete_state.append(discrete_s)
        
        # Convert to single index
        index = 0
        multiplier = 1
        for i in range(len(discrete_state) - 1, -1, -1):
            index += discrete_state[i] * multiplier
            multiplier *= n_bins[i]
        
        return index
    
    # Create agent
    n_states = np.prod(n_bins)
    n_actions = env.action_space.n
    
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42
    )
    
    # Custom training loop for discretized states
    print("Training...")
    for episode in range(1000):
        state, _ = env.reset()
        state = discretize_state(state)
        
        total_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < 500:
            # Get action
            action = agent.get_action(state, training=True)
            
            # Take action
            next_state_raw, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_raw)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done or truncated)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        agent.epsilon_history.append(agent.epsilon)
        agent.decay_epsilon()
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            print(f"Episode {episode + 1}/1000, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    # Evaluate
    print("\nEvaluating...")
    eval_rewards = []
    for _ in range(100):
        state, _ = env.reset()
        state = discretize_state(state)
        
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.get_action(state, training=False)
            next_state_raw, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_raw)
            
            state = next_state
            total_reward += reward
        
        eval_rewards.append(total_reward)
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Max Reward: {np.max(eval_rewards):.0f}")
    
    # Plot results
    agent.plot_training_history()
    
    env.close()
    return agent, env


def demonstrate_trained_agent():
    """Demonstrate a trained agent with visualization."""
    print("\n=== Demonstrating Trained Agent ===\n")
    
    # Load trained Q-learning agent
    env = GridWorldEnv(grid_size=10, render_mode='human')
    agent = QLearningAgent(
        n_states=100,
        n_actions=4
    )
    
    # Try to load saved agent
    try:
        agent.load("results/q_learning_grid_world/agent.pkl")
        print("Loaded trained agent.")
    except:
        print("No saved agent found. Training a new one...")
        agent.train(env, n_episodes=500, verbose=False)
    
    # Run demonstration episodes
    print("\nRunning demonstration episodes...")
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        state, _ = env.reset()
        state_idx = agent._state_to_index(state, env)
        
        total_reward = 0
        steps = 0
        
        for step in range(100):
            # Get action (greedy)
            action = agent.get_action(state_idx, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_idx = agent._state_to_index(next_state, env)
            
            total_reward += reward
            steps += 1
            
            # Render
            env.render()
            
            if terminated:
                print(f"  Goal reached! Steps: {steps}, Reward: {total_reward:.2f}")
                break
            elif truncated:
                print(f"  Episode truncated. Steps: {steps}, Reward: {total_reward:.2f}")
                break
            
            state_idx = next_state_idx
        
        if not (terminated or truncated):
            print(f"  Max steps reached. Reward: {total_reward:.2f}")
    
    env.close()


def main():
    """Main function to run different training scenarios."""
    parser = argparse.ArgumentParser(description='Train RL agents on various environments')
    parser.add_argument('--agent', type=str, default='q_learning',
                       choices=['q_learning', 'dqn', 'both'],
                       help='Which agent to train')
    parser.add_argument('--env', type=str, default='grid_world',
                       choices=['grid_world', 'cartpole', 'both'],
                       help='Which environment to use')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration of trained agent')
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_trained_agent()
        return
    
    # Train based on arguments
    if args.agent == 'q_learning' and args.env == 'grid_world':
        train_q_learning_grid_world()
    elif args.agent == 'dqn' and args.env == 'grid_world':
        train_dqn_grid_world()
    elif args.agent == 'q_learning' and args.env == 'cartpole':
        train_q_learning_cartpole()
    elif args.agent == 'both' and args.env == 'grid_world':
        print("Training both agents on Grid World...\n")
        train_q_learning_grid_world()
        print("\n" + "="*50 + "\n")
        train_dqn_grid_world()
    else:
        print("Training Q-Learning on Grid World (default)...")
        train_q_learning_grid_world()


if __name__ == "__main__":
    main()

