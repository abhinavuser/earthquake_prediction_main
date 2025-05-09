# Earthquake Prediction using Swarm AI and Multi-Agent Reinforcement Learning
## Setup and Implementation Guide

### 1. Environment Setup
First, let's set up our Google Colab environment with all required dependencies:

```python
# Install required packages
!pip install pettingzoo stable-baselines3 pymarl
!pip install geopandas folium scikit-learn pandas numpy matplotlib seaborn
!pip install torch torchvision

# Import necessary libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

# For MARL environment
from pettingzoo import ParallelEnv
from stable_baselines3 import PPO
from typing import Dict, List, Tuple
```

### 2. Data Collection and Preprocessing

```python
class EarthquakeDataProcessor:
    def __init__(self):
        # Download USGS earthquake data (last 30 days as example)
        self.data_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"
        
    def load_data(self):
        # Load earthquake data
        self.df = pd.read_csv(self.data_url)
        return self.df
    
    def preprocess_data(self):
        # Basic preprocessing
        self.df['datetime'] = pd.to_datetime(self.df['time'])
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['day'] = self.df['datetime'].dt.day
        self.df['month'] = self.df['datetime'].dt.month
        
        # Normalize features
        scaler = StandardScaler()
        features = ['latitude', 'longitude', 'depth', 'magnitude']
        self.df[features] = scaler.fit_transform(self.df[features])
        
        return self.df

# Initialize and process data
processor = EarthquakeDataProcessor()
df = processor.load_data()
processed_df = processor.preprocess_data()
```

### 3. MARL Environment Definition

```python
class EarthquakeEnvironment(ParallelEnv):
    def __init__(self, num_agents=5, grid_size=50):
        super().__init__()
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        
        # Action and observation spaces
        self.action_space = {agent: spaces.Box(low=-1, high=1, shape=(4,)) 
                            for agent in self.agents}
        self.observation_space = {agent: spaces.Box(low=-np.inf, high=np.inf, shape=(10,)) 
                                for agent in self.agents}
    
    def reset(self):
        # Reset environment state
        self.current_step = 0
        observations = {agent: self._get_observation() for agent in self.agents}
        return observations
    
    def step(self, actions):
        # Execute one time step within the environment
        rewards = {agent: self._calculate_reward(actions[agent]) for agent in self.agents}
        done = {agent: self.current_step >= self.max_steps for agent in self.agents}
        observations = {agent: self._get_observation() for agent in self.agents}
        info = {agent: {} for agent in self.agents}
        
        self.current_step += 1
        return observations, rewards, done, info
    
    def _get_observation(self):
        # Generate observation for an agent
        return np.random.random(10)  # Placeholder
    
    def _calculate_reward(self, action):
        # Calculate reward based on prediction accuracy
        return 0  # Placeholder
```

### 4. Swarm Intelligence Implementation

```python
class ParticleSwarmOptimization:
    def __init__(self, num_particles, dimensions, bounds):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.particles = self._initialize_particles()
        self.global_best_position = None
        self.global_best_score = float('inf')
    
    def _initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            position = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
            velocity = np.zeros(self.dimensions)
            personal_best_position = position.copy()
            personal_best_score = float('inf')
            particles.append({
                'position': position,
                'velocity': velocity,
                'personal_best_position': personal_best_position,
                'personal_best_score': personal_best_score
            })
        return particles
    
    def optimize(self, objective_function, max_iterations=100):
        for _ in range(max_iterations):
            for particle in self.particles:
                # Evaluate current position
                score = objective_function(particle['position'])
                
                # Update personal best
                if score < particle['personal_best_score']:
                    particle['personal_best_score'] = score
                    particle['personal_best_position'] = particle['position'].copy()
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle['position'].copy()
                
                # Update velocity and position
                w = 0.7  # inertia weight
                c1 = 1.5  # cognitive weight
                c2 = 1.5  # social weight
                
                r1, r2 = np.random.rand(2)
                particle['velocity'] = (w * particle['velocity'] +
                                     c1 * r1 * (particle['personal_best_position'] - particle['position']) +
                                     c2 * r2 * (self.global_best_position - particle['position']))
                
                particle['position'] += particle['velocity']
                
                # Enforce bounds
                particle['position'] = np.clip(particle['position'], self.bounds[0], self.bounds[1])
        
        return self.global_best_position, self.global_best_score
```

### 5. Neural Network Model for Predictions

```python
class EarthquakePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EarthquakePredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

# Initialize model
input_size = 10  # Number of features
hidden_size = 64
output_size = 3  # Magnitude, latitude, longitude
model = EarthquakePredictionModel(input_size, hidden_size, output_size)
```

### 6. Training Pipeline

```python
def train_model(model, train_loader, val_loader, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        
        print(f'Epoch: {epoch+1}, Training Loss: {train_loss/len(train_loader):.6f}, '
              f'Validation Loss: {val_loss/len(val_loader):.6f}')

# Create data loaders and train model
# train_loader = ...
# val_loader = ...
# train_model(model, train_loader, val_loader)
```

### 7. Visualization and Evaluation

```python
def visualize_predictions(true_values, predictions, locations):
    # Create a map centered on the mean location
    center_lat = np.mean(locations[:, 0])
    center_lon = np.mean(locations[:, 1])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    
    # Plot actual earthquakes
    for i in range(len(true_values)):
        folium.Circle(
            location=[locations[i, 0], locations[i, 1]],
            radius=true_values[i] * 1000,  # Scale radius by magnitude
            color='red',
            fill=True,
            popup=f'Actual Magnitude: {true_values[i]:.1f}'
        ).add_to(m)
    
    # Plot predictions
    for i in range(len(predictions)):
        folium.Circle(
            location=[locations[i, 0], locations[i, 1]],
            radius=predictions[i] * 1000,
            color='blue',
            fill=False,
            popup=f'Predicted Magnitude: {predictions[i]:.1f}'
        ).add_to(m)
    
    return m

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predictions.extend(output.numpy())
            true_values.extend(target.numpy())
    
    # Calculate metrics
    mse = np.mean((np.array(predictions) - np.array(true_values)) ** 2)
    mae = np.mean(np.abs(np.array(predictions) - np.array(true_values)))
    
    print(f'Mean Squared Error: {mse:.6f}')
    print(f'Mean Absolute Error: {mae:.6f}')
    
    return predictions, true_values
```

### 8. Main Execution Pipeline

```python
def main():
    # Initialize data processor
    processor = EarthquakeDataProcessor()
    data = processor.load_data()
    processed_data = processor.preprocess_data()
    
    # Initialize MARL environment
    env = EarthquakeEnvironment(num_agents=5)
    
    # Initialize PSO optimizer
    pso = ParticleSwarmOptimization(
        num_particles=20,
        dimensions=model.count_parameters(),
        bounds=(-1, 1)
    )
    
    # Initialize and train model
    model = EarthquakePredictionModel(input_size=10, hidden_size=64, output_size=3)
    
    # Train model
    # train_model(model, train_loader, val_loader)
    
    # Evaluate and visualize results
    # predictions, true_values = evaluate_model(model, test_loader)
    # visualization = visualize_predictions(true_values, predictions, locations)
    
if __name__ == "__main__":
    main()
```

### Usage Instructions:

1. Open Google Colab and create a new notebook
2. Copy each section of code into separate cells
3. Run the cells in order from top to bottom
4. Modify hyperparameters as needed
5. Monitor training progress and visualization outputs

### Key Components:

1. Data Processing: Handles USGS earthquake data
2. MARL Environment: Custom environment for multiple agents
3. PSO Implementation: Swarm intelligence for optimization
4. Neural Network: LSTM-based prediction model
5. Training Pipeline: Complete training workflow
6. Visualization: Geospatial visualization of predictions

### Additional Notes:

- Ensure you have sufficient GPU runtime in Colab
- Save model checkpoints periodically
- Adjust hyperparameters based on performance
- Consider using cross-validation for robust evaluation