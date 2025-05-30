import pandas as pd

# Load parquet file
print("Loading parquet file...")
df = pd.read_parquet('yellow_tripdata_2025-02.parquet')

# Filter for Feb 1-28, 2025 if needed
print("Processing data...")
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df['date'] = df['tpep_pickup_datetime'].dt.date

start_date = pd.to_datetime('2025-02-01').date()
end_date = pd.to_datetime('2025-02-28').date()
filtered_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# Count trips between each OD pair (Origin-Destination pair)
print("Counting trips between each OD pair...")
od_counts = filtered_data.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='trip_count')

# Create a complete OD matrix with all possible combinations
print("Creating complete OD matrix...")
# There are 263 taxi zones (1-263)
all_zones = range(1, 264)

# Create all possible OD pairs
od_pairs = pd.MultiIndex.from_product([all_zones, all_zones], names=['PULocationID', 'DOLocationID'])

# Reindex to include all possible combinations
full_od_matrix = od_counts.set_index(['PULocationID', 'DOLocationID']).reindex(od_pairs, fill_value=0).reset_index()

# Save to CSV
# print("Saving results to CSV...")
# full_od_matrix.to_csv('od_matrix_Feb2025.csv', index=False)

# print(f"Done! Created OD matrix with {len(full_od_matrix)} rows.")
# print(f"Expected rows: {len(all_zones) * len(all_zones)} (263*263 = 69,169)")


################
# Calculate bidirectional flow differences
print("\nCalculating bidirectional flow differences...")

# Define zones to exclude (airports and zone 70)
exclude_zones = [1, 70, 132, 138]

# Convert the full_od_matrix to a pivot table for easier access
flow_matrix = full_od_matrix.pivot(index='PULocationID', columns='DOLocationID', values='trip_count')

# Create a DataFrame to store the flow differences and percentages
flow_diff_results = []

# Create a set to track processed zone pairs
processed_pairs = set()

# Iterate through all zone pairs
for i in all_zones:
    # Skip excluded zones
    if i in exclude_zones:
        continue
        
    for j in all_zones:
        # Skip excluded zones and self-to-self flows
        if j in exclude_zones or i == j:
            continue
            
        # Check if this pair has already been processed (in either direction)
        if (i, j) in processed_pairs or (j, i) in processed_pairs:
            continue
            
        # Mark this pair as processed
        processed_pairs.add((i, j))
        
        flow_i_to_j = flow_matrix.loc[i, j]
        flow_j_to_i = flow_matrix.loc[j, i]
        
        # Only consider flows greater than 100
        if flow_i_to_j <= 100 and flow_j_to_i <= 100:
            continue
        
        # Calculate the flow difference
        if flow_i_to_j > flow_j_to_i:
            # More trips from i to j
            if flow_i_to_j <= 100:  # Skip if flow is not greater than 100
                continue
                
            diff = flow_i_to_j - flow_j_to_i
            percentage = (diff / flow_i_to_j) * 100
            direction = f"{i} to {j}"
            source = i
            target = j
            source_flow = flow_i_to_j
            target_flow = flow_j_to_i
        else:
            # More trips from j to i
            if flow_j_to_i <= 100:  # Skip if flow is not greater than 100
                continue
                
            diff = flow_j_to_i - flow_i_to_j
            percentage = (diff / flow_j_to_i) * 100
            direction = f"{j} to {i}"
            source = j
            target = i
            source_flow = flow_j_to_i
            target_flow = flow_i_to_j
        
        # Store the results
        flow_diff_results.append({
            'PULocationID': source,
            'DOLocationID': target,
            'flow_from_source': source_flow,
            'flow_from_target': target_flow,
            'flow_difference': diff,
            'percentage': percentage,
            'direction': direction
        })

# Convert to DataFrame
flow_diff_df = pd.DataFrame(flow_diff_results)

# Find the top 10 zone pairs with highest percentage difference
top_10_percent = flow_diff_df.sort_values('percentage', ascending=False).head(10)

# Print results
print("\nTop 10 taxi zone pairs with highest percentage of non-returning trips:")
for _, row in top_10_percent.iterrows():
    source = row['PULocationID']
    target = row['DOLocationID']
    diff = row['flow_difference']
    percentage = row['percentage']
    source_flow = row['flow_from_source']
    
    print(f"Net flow from zone {source} to zone {target}: {diff} ({percentage:.2f}% of trips from zone {source} to zone {target} do not return)")
    print(f"  Total trips from zone {source} to zone {target}: {source_flow}")

# Save all results to CSV
flow_diff_df.to_csv('zone_flow_percentages.csv', index=False)
print("\nAll zone pair flow percentages have been saved to 'zone_flow_percentages.csv'")
print(f"Total rows in the output file: {len(flow_diff_df)}")