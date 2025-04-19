import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwarmAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head for actions
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # PSO parameters
        self.velocity = torch.zeros(action_dim)
        self.best_position = None
        self.best_fitness = float('-inf')
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        encoded = self.obs_encoder(obs)
        action_logits = self.policy_head(encoded)
        value = self.value_head(encoded)
        return action_logits, value
    
    def update_position_pso(self, 
                          global_best: torch.Tensor,
                          w: float = 0.7,
                          c1: float = 1.5,
                          c2: float = 1.5):
        """Update agent position using Particle Swarm Optimization"""
        r1, r2 = torch.rand(1), torch.rand(1)
        
        # Update velocity
        self.velocity = (w * self.velocity + 
                        c1 * r1 * (self.best_position - self.get_position()) +
                        c2 * r2 * (global_best - self.get_position()))
        
        # Update position
        new_position = self.get_position() + self.velocity
        self.set_position(new_position)
    
    def get_position(self) -> torch.Tensor:
        """Get current position (weights) of the agent"""
        return torch.cat([p.data.flatten() for p in self.parameters()])
    
    def set_position(self, position: torch.Tensor):
        """Set position (weights) of the agent"""
        start = 0
        for param in self.parameters():
            num_params = param.numel()
            param.data = position[start:start + num_params].reshape(param.shape)
            start += num_params

class SwarmNetwork:
    def __init__(self, 
                 num_agents: int,
                 obs_dim: int,
                 action_dim: int):
        self.agents = [SwarmAgent(obs_dim, action_dim) for _ in range(num_agents)]
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
    
    def update_swarm(self, fitness_scores: List[float]):
        """Update the entire swarm based on fitness scores"""
        for agent, fitness in zip(self.agents, fitness_scores):
            if fitness > agent.best_fitness:
                agent.best_fitness = fitness
                agent.best_position = agent.get_position()
                
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = agent.get_position()
        
        # Update all agents using PSO
        for agent in self.agents:
            agent.update_position_pso(self.global_best_position)