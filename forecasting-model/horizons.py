#!/usr/bin/env python3

import pickle

# Define the horizons variable  
horizons = {
    '1-hour': 1,
    '6-hour': 6,
    '1-day': 24,
    '3-day': 72
}

print("Forecast horizons defined:")
for name, steps in horizons.items():
    print(f"  {name}: {steps} steps")

print(f"\nhorizons variable: {horizons}")
print(f"Type: {type(horizons)}")

# Save to a pickle file so it can be loaded in the notebook
horizons_file = 'notebooks/horizons.pkl'
with open(horizons_file, 'wb') as f:
    pickle.dump(horizons, f)
    
print(f"\nHorizons saved to: {horizons_file}")
