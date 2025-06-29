# File: models/energy/run_residential_sim.py
import argparse
from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ev2gym.simulators import ResidentialEnergySimulator

def main():
    parser = argparse.ArgumentParser(description='Run residential energy simulation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create simulator and run
    sim = ResidentialEnergySimulator(args.config)
    sim.run_simulation()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sim.save_results(str(output_dir))
    print(f"Simulation complete. Results saved to {output_dir.absolute()}")

if __name__ == '__main__':
    main()