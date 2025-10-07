# Reinforcement Learning Assignment

This project implements various reinforcement learning environments and agents for the assignment:
*"Implementuj libovolne prostredi (Pole, Grid, Hra, vlastni ...etc.) pro Reinforcement learning a natrenuj libovolneho agenta (Q-table, DQN, REINFORCE, PPO, DPO ... etc.)"*

## Implemented Components

### Environments

1. **Grid World** (`grid_world.py`)
   - Custom 10x10 grid environment
   - Agent navigates from start to goal
   - Features: walls, danger zones (-5 reward), treasures (+5 reward), goal (+10 reward)
   - Step penalty (-0.1) encourages efficient paths
   - Exploration bonus for visiting new cells

2. **Simple Frozen Lake** (`grid_world.py`)
   - Alternative simpler environment
   - Navigate frozen lake avoiding holes
   - Discrete state and action space

### Agents

1. **Q-Learning Agent** (`q_learning_agent.py`)
   - Tabular Q-learning implementation
   - Epsilon-greedy exploration
   - Suitable for discrete state spaces
   - Features:
     - Q-table visualization
     - Policy visualization
     - Training history plots

2. **Deep Q-Network (DQN) Agent** (`dqn_agent.py`)
   - Neural network-based Q-learning
   - Experience replay buffer
   - Target network for stability
   - Can handle larger/continuous state spaces
   - Features:
     - Customizable network architecture
     - Loss tracking
     - Model checkpointing

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

Train Q-Learning agent on Grid World:
```bash
python train.py --agent q_learning --env grid_world
```

Train DQN agent on Grid World:
```bash
python train.py --agent dqn --env grid_world
```

Train Q-Learning on CartPole (with discretized states):
```bash
python train.py --agent q_learning --env cartpole
```

### Training Both Agents

Compare Q-Learning and DQN on Grid World:
```bash
python train.py --agent both --env grid_world
```

### Demonstration

See a trained agent in action:
```bash
python train.py --demo
```

## Results

Training outputs are saved in the `results/` directory:
- `results/q_learning_grid_world/`
  - `agent.pkl` - Saved Q-table and parameters
  - `history.json` - Training history and evaluation metrics
- `results/dqn_grid_world/`
  - `model.pth` - Saved neural network weights
  - `history.json` - Training history and metrics

## Visualizations

The training script automatically generates several visualizations:

1. **Training Progress**
   - Episode rewards over time
   - Moving average of rewards
   - Episode lengths
   - Exploration rate (epsilon) decay

2. **Q-Learning Specific**
   - Q-table heatmap
   - Learned policy visualization (arrows showing best action per state)

3. **DQN Specific**
   - Training loss over time
   - Network convergence metrics

## Customization

### Create Your Own Environment

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyCustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,))
        
    def reset(self, seed=None):
        # Reset environment to initial state
        return observation, info
        
    def step(self, action):
        # Execute action and return results
        return observation, reward, terminated, truncated, info
```

### Modify Agent Parameters

```python
# Q-Learning hyperparameters
agent = QLearningAgent(
    n_states=100,
    n_actions=4,
    learning_rate=0.1,        # How fast the agent learns
    discount_factor=0.99,     # How much to value future rewards
    epsilon=1.0,              # Initial exploration rate
    epsilon_decay=0.995,      # How fast to reduce exploration
    epsilon_min=0.01          # Minimum exploration rate
)

# DQN hyperparameters
agent = DQNAgent(
    state_size=2,
    action_size=4,
    learning_rate=0.001,      # Lower for neural networks
    buffer_size=10000,        # Experience replay buffer size
    batch_size=32,            # Batch size for training
    hidden_size=128,          # Neural network hidden layer size
    update_target_every=100   # Steps between target network updates
)
```

## Algorithm Details

### Q-Learning
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- **Advantages**: Simple, guaranteed convergence for tabular case
- **Limitations**: Requires discrete state space or discretization

### Deep Q-Network (DQN)
- **Key Features**:
  - Neural network approximates Q-function
  - Experience replay breaks correlation
  - Target network provides stability
- **Advantages**: Handles large/continuous state spaces
- **Limitations**: More complex, requires tuning

## Extending the Project

1. **Add New Environments**:
   - Mountain Car
   - Lunar Lander
   - Custom game environments

2. **Implement More Algorithms**:
   - REINFORCE (policy gradient)
   - PPO (Proximal Policy Optimization)
   - A2C/A3C (Actor-Critic)
   - SAC (Soft Actor-Critic)

3. **Advanced Features**:
   - Prioritized experience replay
   - Dueling DQN
   - Double DQN
   - Rainbow DQN

## References

- Sutton & Barto - "Reinforcement Learning: An Introduction"
- DQN Paper: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- OpenAI Gymnasium Documentation: https://gymnasium.farama.org/

