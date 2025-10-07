"""
Grid World Environment for Reinforcement Learning

A simple grid world where an agent needs to navigate from start to goal
while avoiding obstacles and collecting rewards.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    Grid World Environment
    
    The agent starts at a random position and needs to reach the goal.
    - Empty cells: 0
    - Walls: -1
    - Goal: +10
    - Danger zones: -5
    - Treasure: +5
    
    Actions:
    - 0: UP
    - 1: RIGHT
    - 2: DOWN
    - 3: LEFT
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, grid_size: int = 10, render_mode: Optional[str] = None):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, 
            high=grid_size-1, 
            shape=(2,), 
            dtype=np.int32
        )
        
        # Action mappings
        self._action_to_direction = {
            0: (-1, 0),  # UP
            1: (0, 1),   # RIGHT
            2: (1, 0),   # DOWN
            3: (0, -1),  # LEFT
        }
        
        # Initialize grid
        self._create_grid()
        
        # Rendering
        self.window = None
        self.clock = None
        
    def _create_grid(self):
        """Create the grid world with obstacles and rewards."""
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # Add walls (obstacles)
        # Vertical wall
        self.grid[2:7, 3] = -1
        # Horizontal wall
        self.grid[5, 5:9] = -1
        
        # Add danger zones
        self.grid[7:9, 7:9] = -5
        self.grid[1:3, 7:9] = -5
        
        # Add treasures
        self.grid[0, 9] = 5
        self.grid[8, 1] = 5
        
        # Set goal position
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)
        self.grid[self.goal_pos] = 10
        
        # Valid starting positions (not on walls or goal)
        self.valid_starts = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 0:
                    self.valid_starts.append((i, j))
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Random starting position
        start_idx = self.np_random.integers(0, len(self.valid_starts))
        self.agent_pos = list(self.valid_starts[start_idx])
        
        # Track visited positions for exploration bonus
        self.visited = set()
        self.visited.add(tuple(self.agent_pos))
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step within the environment."""
        # Get direction
        direction = self._action_to_direction[action]
        
        # Calculate new position
        new_pos = [
            self.agent_pos[0] + direction[0],
            self.agent_pos[1] + direction[1]
        ]
        
        # Check boundaries
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            
            # Check if not a wall
            if self.grid[new_pos[0], new_pos[1]] != -1:
                self.agent_pos = new_pos
                reward = -0.1  # Small penalty for each step
            else:
                reward = -1  # Penalty for hitting wall
        else:
            reward = -1  # Penalty for trying to leave grid
        
        # Get reward from current position
        current_cell_reward = self.grid[self.agent_pos[0], self.agent_pos[1]]
        if current_cell_reward != -1:  # Not a wall
            reward += current_cell_reward
        
        # Exploration bonus
        if tuple(self.agent_pos) not in self.visited:
            reward += 0.1
            self.visited.add(tuple(self.agent_pos))
        
        # Check if goal reached
        terminated = (tuple(self.agent_pos) == self.goal_pos)
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return np.array(self.agent_pos, dtype=np.int32)
    
    def _get_info(self) -> dict:
        """Get additional information."""
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "distance_to_goal": np.linalg.norm(
                np.array(self.agent_pos) - np.array(self.goal_pos)
            )
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """Render one frame of the environment."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create color map
        display_grid = np.zeros((self.grid_size, self.grid_size, 3))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == -1:  # Wall
                    display_grid[i, j] = [0.2, 0.2, 0.2]
                elif self.grid[i, j] == -5:  # Danger
                    display_grid[i, j] = [1.0, 0.2, 0.2]
                elif self.grid[i, j] == 5:  # Treasure
                    display_grid[i, j] = [1.0, 1.0, 0.0]
                elif (i, j) == self.goal_pos:  # Goal
                    display_grid[i, j] = [0.0, 1.0, 0.0]
                else:  # Empty
                    display_grid[i, j] = [0.9, 0.9, 0.9]
        
        # Mark visited cells
        for pos in self.visited:
            if self.grid[pos[0], pos[1]] == 0:
                display_grid[pos[0], pos[1]] = [0.7, 0.8, 0.9]
        
        # Mark agent position
        display_grid[self.agent_pos[0], self.agent_pos[1]] = [0.0, 0.0, 1.0]
        
        ax.imshow(display_grid)
        ax.set_title(f"Grid World - Agent at {self.agent_pos}")
        ax.grid(True, which='both', color='black', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        
        # Add grid coordinates
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] not in [0, -1]:
                    ax.text(j, i, f'{int(self.grid[i, j])}', 
                           ha='center', va='center', color='white', fontsize=12)
        
        plt.tight_layout()
        
        if self.render_mode == "human":
            plt.show()
        else:
            # Convert to RGB array
            fig.canvas.draw()
            rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return rgb_array
    
    def close(self):
        """Close the environment."""
        if self.window is not None:
            plt.close('all')


# Alternative simpler environment: Frozen Lake
class SimpleFrozenLake(gym.Env):
    """
    Simplified Frozen Lake Environment
    
    S: Start
    F: Frozen (safe)
    H: Hole (danger)
    G: Goal
    
    The agent must navigate from S to G without falling into holes.
    """
    
    def __init__(self, map_size: int = 4, slippery: bool = False):
        super().__init__()
        
        self.map_size = map_size
        self.slippery = slippery
        
        # Create map
        self.desc = self._generate_random_map()
        self.nrow, self.ncol = self.desc.shape
        self.nA = 4  # Number of actions
        self.nS = self.nrow * self.ncol  # Number of states
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
        # Initial state distribution
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[0] = 1.0
        
    def _generate_random_map(self) -> np.ndarray:
        """Generate a random map."""
        desc = np.array([['S', 'F', 'F', 'F'],
                        ['F', 'H', 'F', 'H'],
                        ['F', 'F', 'F', 'H'],
                        ['H', 'F', 'F', 'G']], dtype='<U1')
        return desc
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset to initial state."""
        super().reset(seed=seed)
        self.s = 0  # Start position
        return int(self.s), {}
    
    def step(self, action: int):
        """Take a step in the environment."""
        row = self.s // self.ncol
        col = self.s % self.ncol
        
        # Movement
        if action == 0:  # UP
            row = max(0, row - 1)
        elif action == 1:  # RIGHT
            col = min(self.ncol - 1, col + 1)
        elif action == 2:  # DOWN
            row = min(self.nrow - 1, row + 1)
        elif action == 3:  # LEFT
            col = max(0, col - 1)
        
        self.s = row * self.ncol + col
        
        # Get cell type
        cell = self.desc[row, col]
        
        # Determine reward and termination
        if cell == 'G':
            reward = 1.0
            terminated = True
        elif cell == 'H':
            reward = -1.0
            terminated = True
        else:
            reward = -0.01  # Small penalty for each step
            terminated = False
        
        return int(self.s), reward, terminated, False, {}
    
    def render(self):
        """Simple text rendering."""
        row = self.s // self.ncol
        col = self.s % self.ncol
        
        desc = self.desc.copy()
        desc[row, col] = 'X'  # Mark agent position
        
        print("\nFrozen Lake:")
        print(desc)
        print()

