import pandas as pd
import numpy as np
from typing import Tuple, List
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

class SeismicDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_and_preprocess(self, 
                           data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess seismic data"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Extract features
        features = self._extract_features(df)
        
        # Extract targets (future earthquake occurrences)
        targets = self._extract_targets(df)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        return scaled_features, targets
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract relevant features from seismic data"""
        features = []
        
        # Time-based features
        df['timestamp'] = pd.to_datetime(df['time'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Location features
        features.extend([
            df['latitude'].values,
            df['longitude'].values,
            df['depth'].values,
        ])
        
        # Historical features
        features.extend([
            self._calculate_rolling_mean(df['magnitude'], window=24),
            self._calculate_rolling_std(df['magnitude'], window=24),
        ])
        
        return np.column_stack(features)
    
    def _extract_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Extract target variables (future earthquake occurrences)"""
        # Define what constitutes a significant earthquake (e.g., magnitude > 4.0)
        significant_threshold = 4.0
        prediction_window = 24  # hours
        
        targets = []
        for i in range(len(df) - prediction_window):
            future_window = df['magnitude'].iloc[i:i + prediction_window]
            targets.append(any(future_window > significant_threshold))
        
        # Pad the last prediction_window entries
        targets.extend([False] * prediction_window)
        return np.array(targets)
    
    @staticmethod
    def _calculate_rolling_mean(series: pd.Series, 
                              window: int) -> np.ndarray:
        """Calculate rolling mean with handling for NaN values"""
        return series.rolling(window=window, min_periods=1).mean().fillna(0)
    
    @staticmethod
    def _calculate_rolling_std(series: pd.Series, 
                             window: int) -> np.ndarray:
        """Calculate rolling standard deviation with handling for NaN values"""
        return series.rolling(window=window, min_periods=1).std().fillna(0)