import numpy as np
import gym
from pettingzoo import ParallelEnv
from typing import Dict, List, Tuple

class EarthquakeEnv(ParallelEnv):
    def __init__(self, num_agents: int, grid_size: int = 50):
        super().__init__()
        self.num_agents = num_agents
        self.grid_size = grid_size
        
        # Define action and observation spaces
        self.action_spaces = {
            f"agent_{i}": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            for i in range(num_agents)
        }
        
        # Observation space includes: position, local seismic features, neighboring agents' data
        self.observation_spaces = {
            f"agent_{i}": gym.spaces.Dict({
                "position": gym.spaces.Box(low=0, high=grid_size, shape=(2,), dtype=np.float32),
                "seismic_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                "neighbor_data": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
            }) for i in range(num_agents)
        }

    def reset(self) -> Dict:
        """Reset environment to initial state"""
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_positions = {
            agent: np.random.uniform(0, self.grid_size, size=(2,))
            for agent in self.agents
        }
        return self._get_observations()

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one time step within the environment"""
        # Update agent positions based on actions
        for agent_id, action in actions.items():
            self.agent_positions[agent_id] += action
            self.agent_positions[agent_id] = np.clip(
                self.agent_positions[agent_id], 0, self.grid_size
            )
        
        observations = self._get_observations()
        rewards = self._compute_rewards(actions)
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, dones, infos

    def _get_observations(self) -> Dict:
        """Generate observations for all agents"""
        observations = {}
        for agent in self.agents:
            observations[agent] = {
                "position": self.agent_positions[agent],
                "seismic_features": self._get_seismic_features(agent),
                "neighbor_data": self._get_neighbor_data(agent)
            }
        return observations

    def _get_seismic_features(self, agent: str) -> np.ndarray:
        """Simulate seismic features for given agent position"""
        position = self.agent_positions[agent]
        # Placeholder for actual seismic data processing
        return np.random.normal(0, 1, size=(4,))

    def _get_neighbor_data(self, agent: str) -> np.ndarray:
        """Get data from neighboring agents"""
        current_pos = self.agent_positions[agent]
        neighbor_data = []
        for other_agent in self.agents:
            if other_agent != agent:
                other_pos = self.agent_positions[other_agent]
                distance = np.linalg.norm(current_pos - other_pos)
                neighbor_data.extend([distance, *other_pos])
        return np.array(neighbor_data[:8])  # Limit to nearest neighbors

    def _compute_rewards(self, actions: Dict) -> Dict:
        """Compute rewards for all agents"""
        rewards = {}
        for agent in self.agents:
            # Reward based on prediction accuracy and collaboration
            rewards[agent] = self._calculate_prediction_reward(agent)
        return rewards

    def _calculate_prediction_reward(self, agent: str) -> float:
        """Calculate reward based on prediction accuracy"""
        # Placeholder for actual reward calculation
        return np.random.uniform(-1, 1)