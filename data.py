import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_seismic_data(num_records=1000):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate timestamps
    base_date = datetime(2024, 1, 1)
    timestamps = [base_date + timedelta(hours=i) for i in range(num_records)]
    
    # Generate sample data
    data = {
        'time': timestamps,
        'latitude': np.random.uniform(25.0, 45.0, num_records),  # Sample region
        'longitude': np.random.uniform(120.0, 150.0, num_records),
        'depth': np.random.uniform(0, 100, num_records),
        'magnitude': np.random.normal(3.0, 1.0, num_records)  # Most earthquakes are small
    }
    
    # Create some significant earthquakes (magnitude > 4.0)
    significant_indices = np.random.choice(num_records, size=int(num_records * 0.1))
    data['magnitude'][significant_indices] = np.random.normal(5.0, 0.5, len(significant_indices))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by time
    df = df.sort_values('time')
    
    # Save to CSV
    df.to_csv('seismic_data.csv', index=False)
    print("Sample seismic data has been generated and saved to 'seismic_data.csv'")
    print("\nFirst few records:")
    print(df.head())

if __name__ == "__main__":
    generate_sample_seismic_data()