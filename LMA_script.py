# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:45:40 2023
Edited with enhanced visualization options

@author: lealn
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # For 3D visualization

# Load LMA data
filename = 'C:/Users/Werverton/Documents/LMA-mestrado/data/LL_240726_195000_0600.dat'
headerlength = 55  # Adjusted based on your file structure

# Load data with error handling
try:
    sfm, lat, lon, alt, Xisq, nstn, dBW = np.genfromtxt(
        filename,
        dtype=(float, float, float, float, float, int, float),
        unpack=True,
        skip_header=headerlength,
        comments="#",
        usecols=[0, 1, 2, 3, 4, 5, 6]
    )
    data = {'sfm': sfm, 'lat': lat, 'lon': lon, 'alt': alt, 'Xisq': Xisq, 'nstn': nstn, 'dBW': dBW}
    df = pd.DataFrame(data)
    
    # Filter based on quality criteria
    filtered_df = df.query('nstn > 6')
    print(f"Loaded {len(filtered_df)} events after filtering")
    
    # Convert time format
    m, s = divmod(filtered_df['sfm'], 60)
    h, m = divmod(m, 60)
    
except Exception as e:
    print(f"Error loading LMA data: {str(e)}")
    exit()

# ==================================================================
# Enhanced Visualization Section
# ==================================================================

def plot_basic_stats():
    """Plot basic statistics of the LMA data"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Altitude vs Time with station count
    plt.subplot(2, 2, 1)
    sc = plt.scatter(s, filtered_df['alt']/1000, 
                    c=filtered_df['nstn'], 
                    s=10, 
                    cmap='viridis',
                    alpha=0.7)
    plt.colorbar(sc, label='Number of Stations')
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (km)")
    plt.title("Event Altitude vs Time")
    plt.grid(True)

    # Plot 2: Geographic distribution
    plt.subplot(2, 2, 2)
    sc = plt.scatter(filtered_df['lon'], filtered_df['lat'], 
                    c=filtered_df['alt']/1000, 
                    s=5, 
                    cmap='jet',
                    alpha=0.5)
    plt.colorbar(sc, label='Altitude (km)')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographic Distribution")

    # Plot 3: Quality metrics
    plt.subplot(2, 2, 3)
    plt.hist(filtered_df['Xisq'], bins=50, color='skyblue', edgecolor='black')
    plt.xlabel("Reduced Chi-Square")
    plt.ylabel("Count")
    plt.title("Quality Metric Distribution")
    plt.axvline(x=1.0, color='red', linestyle='--')

    # Plot 4: Power distribution
    plt.subplot(2, 2, 4)
    plt.hist(filtered_df['dBW'], bins=50, color='salmon', edgecolor='black')
    plt.xlabel("Power (dBW)")
    plt.ylabel("Count")
    plt.title("Power Distribution")

    plt.tight_layout()
    plt.show()

def plot_3d_trajectories():
    """Create 3D visualization of lightning events"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize altitude for coloring
    norm_alt = (filtered_df['alt'] - filtered_df['alt'].min()) / (filtered_df['alt'].max() - filtered_df['alt'].min())
    
    sc = ax.scatter(
        filtered_df['lon'],
        filtered_df['lat'],
        filtered_df['alt']/1000,
        c=norm_alt,
        cmap='plasma',
        s=5,
        alpha=0.7
    )
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude (km)')
    ax.set_title('3D Lightning Event Distribution')
    fig.colorbar(sc, label='Normalized Altitude')
    plt.show()

def plot_time_series_analysis():
    """Analyze temporal evolution of events"""
    plt.figure(figsize=(15, 5))
    
    # Calculate event rate
    time_bins = np.arange(s.min(), s.max(), 1)  # 1-second bins
    event_counts, _ = np.histogram(s, bins=time_bins)
    
    plt.subplot(1, 2, 1)
    plt.plot(time_bins[:-1], event_counts, 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('Events per second')
    plt.title('Lightning Event Rate')
    plt.grid(True)
    
    # Altitude distribution over time
    plt.subplot(1, 2, 2)
    plt.hexbin(s, filtered_df['alt']/1000, 
               gridsize=50, 
               cmap='inferno', 
               bins='log')
    plt.colorbar(label='Log10(Count)')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    plt.title('Altitude-Time Distribution')
    
    plt.tight_layout()
    plt.show()

# ==================================================================
# Execute visualizations
# ==================================================================
if not filtered_df.empty:
    plot_basic_stats()
    plot_3d_trajectories()
    plot_time_series_analysis()
    
    # Save filtered data for further analysis
    filtered_df.to_csv('filtered_lma_events.csv', index=False)
    print("Saved filtered data to 'filtered_lma_events.csv'")
else:
    print("No events remaining after filtering - check your data quality thresholds")

# Note: The E-field and GLM visualization sections remain commented as in your original script
# Uncomment and adapt them when you have those data files available