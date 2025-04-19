import os
from environment.data import generate_sample_seismic_data
from environment.train import main as train_main

def setup_project():
    print("Setting up Earthquake Prediction Project...")
    
    # Check if seismic data exists
    if not os.path.exists('seismic_data.csv'):
        print("\nGenerating sample seismic data...")
        generate_sample_seismic_data()
    else:
        print("\nSeismic data file already exists.")
    
    print("\nStarting training process...")
    train_main()

if __name__ == "__main__":
    setup_project()