import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import numpy as np

# Load data
print("Loading parquet files...")
yellow_df = pd.read_parquet('yellow_tripdata_2025-02.parquet')
green_df = pd.read_parquet('green_tripdata_2025-02.parquet')

# Load taxi_zones.csv file to get the relationship between zone and borough
zones_df = pd.read_csv('taxi_zones.csv')

# Process Yellow taxi data
# Convert datetime columns to date format
yellow_df['pickup_date'] = pd.to_datetime(yellow_df['tpep_pickup_datetime']).dt.date
yellow_df['dropoff_date'] = pd.to_datetime(yellow_df['tpep_dropoff_datetime']).dt.date

# Merge data to add borough information for pickup locations
yellow_pickup = yellow_df.merge(zones_df[['LocationID', 'borough']], 
                    left_on='PULocationID', 
                    right_on='LocationID', 
                    how='left')
yellow_pickup = yellow_pickup.rename(columns={'borough': 'pickup_borough'})

# Merge data to add borough information for dropoff locations
yellow_dropoff = yellow_df.merge(zones_df[['LocationID', 'borough']], 
                     left_on='DOLocationID', 
                     right_on='LocationID', 
                     how='left')
yellow_dropoff = yellow_dropoff.rename(columns={'borough': 'dropoff_borough'})

# Process Green taxi data
# Convert datetime columns to date format
green_df['pickup_date'] = pd.to_datetime(green_df['lpep_pickup_datetime']).dt.date
green_df['dropoff_date'] = pd.to_datetime(green_df['lpep_dropoff_datetime']).dt.date

# Merge data to add borough information for pickup locations
green_pickup = green_df.merge(zones_df[['LocationID', 'borough']], 
                    left_on='PULocationID', 
                    right_on='LocationID', 
                    how='left')
green_pickup = green_pickup.rename(columns={'borough': 'pickup_borough'})

# Merge data to add borough information for dropoff locations
green_dropoff = green_df.merge(zones_df[['LocationID', 'borough']], 
                     left_on='DOLocationID', 
                     right_on='LocationID', 
                     how='left')
green_dropoff = green_dropoff.rename(columns={'borough': 'dropoff_borough'})

# Count pickups by date and borough for yellow taxis
yellow_pickup_counts = yellow_pickup.groupby(['pickup_date', 'pickup_borough']).size().reset_index(name='pickup_count')

# Count dropoffs by date and borough for yellow taxis
yellow_dropoff_counts = yellow_dropoff.groupby(['dropoff_date', 'dropoff_borough']).size().reset_index(name='dropoff_count')

# Count pickups by date and borough for green taxis
green_pickup_counts = green_pickup.groupby(['pickup_date', 'pickup_borough']).size().reset_index(name='pickup_count')

# Count dropoffs by date and borough for green taxis
green_dropoff_counts = green_dropoff.groupby(['dropoff_date', 'dropoff_borough']).size().reset_index(name='dropoff_count')

# Combine yellow and green taxi data
# For pickups
pickup_counts = pd.concat([yellow_pickup_counts, green_pickup_counts])
pickup_counts = pickup_counts.groupby(['pickup_date', 'pickup_borough']).sum().reset_index()

# For dropoffs
dropoff_counts = pd.concat([yellow_dropoff_counts, green_dropoff_counts])
dropoff_counts = dropoff_counts.groupby(['dropoff_date', 'dropoff_borough']).sum().reset_index()

# Create date list (February 1 to February 28)
date_range = pd.date_range(start='2025-02-01', end='2025-02-28')
date_list = [date.date() for date in date_range]

# Get list of all boroughs
boroughs = zones_df['borough'].unique()

# Create directory to save charts
output_dir = 'charts'
os.makedirs(output_dir, exist_ok=True)

# Set chart style
plt.style.use('ggplot')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Increase font size for all plots
plt.rcParams.update({'font.size': 16})

# Create a list of dates for the vertical lines (Sunday-Monday transitions)
# February 2025: 1st is Saturday, so 2nd is Sunday
# We need lines between Sunday and Monday: 3-4, 10-11, 17-18, 24-25
week_dividers = [datetime.strptime('2025-02-04', '%Y-%m-%d').date(),
                datetime.strptime('2025-02-11', '%Y-%m-%d').date(),
                datetime.strptime('2025-02-18', '%Y-%m-%d').date(),
                datetime.strptime('2025-02-25', '%Y-%m-%d').date()]

# Group 1: Pickup counts for all boroughs (dual y-axis)
fig, ax1 = plt.subplots(figsize=(14, 8))

# Create a second y-axis
ax2 = ax1.twinx()

# Plot Manhattan on the left axis
manhattan_data = pickup_counts[pickup_counts['pickup_borough'] == 'Manhattan']
full_manhattan_data = pd.DataFrame({'pickup_date': date_list})
full_manhattan_data = full_manhattan_data.merge(manhattan_data, on='pickup_date', how='left').fillna(0)
ax1.plot(full_manhattan_data['pickup_date'], full_manhattan_data['pickup_count'], 
         label='Manhattan', color='#1f77b4', marker='o', markersize=6, linewidth=2)

# Plot other boroughs on the right axis
other_boroughs = [b for b in boroughs if b != 'Manhattan']
for i, borough in enumerate(other_boroughs):
    borough_data = pickup_counts[pickup_counts['pickup_borough'] == borough]
    full_date_data = pd.DataFrame({'pickup_date': date_list})
    full_date_data = full_date_data.merge(borough_data, on='pickup_date', how='left').fillna(0)
    ax2.plot(full_date_data['pickup_date'], full_date_data['pickup_count'], 
             label=borough, color=colors[(i+1) % len(colors)], marker='o', markersize=6, linewidth=2)

# Add vertical lines for week divisions (Sunday-Monday transitions)
for date in week_dividers:
    ax1.axvline(x=date, color='black', linestyle='--', alpha=0.7)

# Set labels and title
ax1.set_title('Daily Pickup Counts for Each Borough', fontsize=18)
ax1.set_xlabel('Date', fontsize=16)
ax1.set_ylabel('Pickup Count (Manhattan)', fontsize=16, color='#1f77b4')
ax2.set_ylabel('Pickup Count (Other Boroughs)', fontsize=16, color='#ff7f0e')

# Set x-axis format
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))

# Set x-axis limits to show only February 1-28
ax1.set_xlim([date_list[0], date_list[-1]])

# Rotate x-axis labels
plt.xticks(rotation=45)

# Add legends for both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)

# Enable grid
ax1.grid(True)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig(f'{output_dir}/all_boroughs_pickups.png')
plt.close()

# Group 1: Dropoff counts for all boroughs (dual y-axis)
fig, ax1 = plt.subplots(figsize=(14, 8))

# Create a second y-axis
ax2 = ax1.twinx()

# Plot Manhattan on the left axis
manhattan_data = dropoff_counts[dropoff_counts['dropoff_borough'] == 'Manhattan']
full_manhattan_data = pd.DataFrame({'dropoff_date': date_list})
full_manhattan_data = full_manhattan_data.merge(manhattan_data, on='dropoff_date', how='left').fillna(0)
ax1.plot(full_manhattan_data['dropoff_date'], full_manhattan_data['dropoff_count'], 
         label='Manhattan', color='#1f77b4', marker='o', markersize=6, linewidth=2)

# Plot other boroughs on the right axis
other_boroughs = [b for b in boroughs if b != 'Manhattan']
for i, borough in enumerate(other_boroughs):
    borough_data = dropoff_counts[dropoff_counts['dropoff_borough'] == borough]
    full_date_data = pd.DataFrame({'dropoff_date': date_list})
    full_date_data = full_date_data.merge(borough_data, on='dropoff_date', how='left').fillna(0)
    ax2.plot(full_date_data['dropoff_date'], full_date_data['dropoff_count'], 
             label=borough, color=colors[(i+1) % len(colors)], marker='o', markersize=6, linewidth=2)

# Add vertical lines for week divisions (Sunday-Monday transitions)
for date in week_dividers:
    ax1.axvline(x=date, color='black', linestyle='--', alpha=0.7)

# Set labels and title
ax1.set_title('Daily Dropoff Counts for Each Borough', fontsize=18)
ax1.set_xlabel('Date', fontsize=16)
ax1.set_ylabel('Dropoff Count (Manhattan)', fontsize=16, color='#1f77b4')
ax2.set_ylabel('Dropoff Count (Other Boroughs)', fontsize=16, color='#ff7f0e')

# Set x-axis format
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))

# Set x-axis limits to show only February 1-28
ax1.set_xlim([date_list[0], date_list[-1]])

# Rotate x-axis labels
plt.xticks(rotation=45)

# Add legends for both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)

# Enable grid
ax1.grid(True)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig(f'{output_dir}/all_boroughs_dropoffs.png')
plt.close()

# Group 2: Pickup and dropoff counts for each borough
for i, borough in enumerate(boroughs):
    plt.figure(figsize=(14, 8))
    
    # Get pickup data for this borough
    pickup_data = pickup_counts[pickup_counts['pickup_borough'] == borough]
    full_pickup_data = pd.DataFrame({'pickup_date': date_list})
    full_pickup_data = full_pickup_data.merge(pickup_data, on='pickup_date', how='left').fillna(0)
    
    # Get dropoff data for this borough
    dropoff_data = dropoff_counts[dropoff_counts['dropoff_borough'] == borough]
    full_dropoff_data = pd.DataFrame({'dropoff_date': date_list})
    full_dropoff_data = full_dropoff_data.merge(dropoff_data, on='dropoff_date', how='left').fillna(0)
    
    # Plot pickup line
    plt.plot(full_pickup_data['pickup_date'], full_pickup_data['pickup_count'], 
             label='Pickups', color='#1f77b4', marker='o', markersize=6, linewidth=2)
    
    # Plot dropoff line
    plt.plot(full_dropoff_data['dropoff_date'], full_dropoff_data['dropoff_count'], 
             label='Dropoffs', color='#ff7f0e', marker='o', markersize=6, linewidth=2)
    
    # Add vertical lines for week divisions (Sunday-Monday transitions)
    for date in week_dividers:
        plt.axvline(x=date, color='black', linestyle='--', alpha=0.7)
    
    # Set labels and title
    plt.title(f'Daily Pickup and Dropoff Counts for {borough} Borough', fontsize=18)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    
    # Add legend
    plt.legend(fontsize=14)
    
    # Enable grid
    plt.grid(True)
    
    # Set x-axis format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    
    # Set x-axis limits to show only February 1-28
    plt.xlim([date_list[0], date_list[-1]])
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'{output_dir}/{borough}_pickups_dropoffs.png')
    plt.close()

print("All charts have been generated and saved to the 'charts' directory")