import numpy as np
import torch
from torch.optim import Adam
from typing import List, Tuple
from environment import EarthquakeEnv
from swarm_agent import SwarmNetwork
from environment.data_processor import SeismicDataProcessor

class Trainer:
    def __init__(self,
                 num_agents: int,
                 obs_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4):
        self.env = EarthquakeEnv(num_agents)
        self.swarm_network = SwarmNetwork(num_agents, obs_dim, action_dim)
        self.optimizers = [
            Adam(agent.parameters(), lr=learning_rate)
            for agent in self.swarm_network.agents
        ]
        
    def train_episode(self, 
                     max_steps: int = 1000) -> Tuple[float, List[float]]:
        """Train for one episode"""
        observations = self.env.reset()
        episode_rewards = []
        swarm_fitness_scores = [0] * len(self.swarm_network.agents)
        
        for step in range(max_steps):
            # Get actions from all agents
            actions = {}
            for idx, agent in enumerate(self.swarm_network.agents):
                obs = torch.FloatTensor(self._process_observation(
                    observations[f"agent_{idx}"]))
                action_logits, value = agent(obs)
                action = torch.tanh(action_logits).detach().numpy()
                actions[f"agent_{idx}"] = action
            
            # Step environment
            new_observations, rewards, dones, _ = self.env.step(actions)
            
            # Update fitness scores
            for idx, reward in rewards.items():
                agent_idx = int(idx.split('_')[1])
                swarm_fitness_scores[agent_idx] += reward
            
            episode_rewards.append(sum(rewards.values()))
            observations = new_observations
            
            if all(dones.values()):
                break
        
        # Update swarm based on episode performance
        self.swarm_network.update_swarm(swarm_fitness_scores)
        
        return sum(episode_rewards), swarm_fitness_scores
    
    def _process_observation(self, obs: dict) -> np.ndarray:
        """Process observation dictionary into flat array"""
        return np.concatenate([
            obs['position'],
            obs['seismic_features'],
            obs['neighbor_data']
        ])

def main():
    # Initialize data processor and load data
    data_processor = SeismicDataProcessor()
    features, targets = data_processor.load_and_preprocess('seismic_data.csv')
    
    # Training parameters
    num_agents = 10
    obs_dim = 14  # position(2) + seismic_features(4) + neighbor_data(8)
    action_dim = 2
    num_episodes = 1000
    
    # Initialize trainer
    trainer = Trainer(num_agents, obs_dim, action_dim)
    
    # Training loop
    for episode in range(num_episodes):
        total_reward, fitness_scores = trainer.train_episode()
        
        if episode % 10 == 0:
            print(f"Episode {episode}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Average Agent Fitness: {np.mean(fitness_scores):.2f}")
            print("-" * 50)

if __name__ == "__main__":
    main()