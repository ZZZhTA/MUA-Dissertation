import pandas as pd

# Load parquet file
df = pd.read_parquet('yellow_tripdata_2025-02.parquet')

df.head()

# Load your DataFrame (replace with your actual data loading code)
# df = pd.read_csv('your_taxi_data.csv')

# 1. Data Preparation
# Convert to datetime and extract relevant time components
# df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
# df['date'] = df['tpep_pickup_datetime'].dt.date
# df['hour'] = df['tpep_pickup_datetime'].dt.hour
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df['date'] = df['tpep_dropoff_datetime'].dt.date
df['hour'] = df['tpep_dropoff_datetime'].dt.hour

# 2. Filter for Feb 1-28, 2025
start_date = pd.to_datetime('2025-02-01').date()
end_date = pd.to_datetime('2025-02-28').date()
july_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# 3. Create hourly time bins (336 time points)
all_hours = pd.date_range(
    start='2025-02-01 00:00:00',
    end='2025-02-28 23:00:00',
    freq='H'
).strftime('%Y-%m-%d %H:00')

# 4. Aggregate pickups by zone and hour
# hourly_pickups = july_data.groupby(
#     ['PULocationID', pd.Grouper(key='tpep_pickup_datetime', freq='H')]
# ).size().reset_index(name='pickups')
hourly_pickups = july_data.groupby(
    ['PULocationID', pd.Grouper(key='tpep_dropoff_datetime', freq='H')]
).size().reset_index(name='dropoffs')

# 5. Create pivot table with all zones and all hours
# First create complete multi-index
all_zones = range(1, 264)
full_index = pd.MultiIndex.from_product(
    [all_zones, all_hours],
    names=['PULocationID', 'hour']
)

# Reindex to include all possible combinations
# hourly_pickups_complete = hourly_pickups.set_index(
#     ['PULocationID', hourly_pickups['tpep_pickup_datetime'].dt.strftime('%Y-%m-%d %H:00')]
# ).reindex(full_index, fill_value=0).reset_index()
hourly_pickups_complete = hourly_pickups.set_index(
    ['PULocationID', hourly_pickups['tpep_dropoff_datetime'].dt.strftime('%Y-%m-%d %H:00')]
).reindex(full_index, fill_value=0).reset_index()

# 6. Pivot to wide format (263 zones × 672 hours)
pivot_table = hourly_pickups_complete.pivot(
    index='PULocationID',
    columns='hour',
    # values='pickups'
    values='dropoffs'
)

# 7. Ensure all 263 zones are present (some might have zero pickups all days)
pivot_table = pivot_table.reindex(all_zones, fill_value=0)

# 8. Sort zones and columns
pivot_table = pivot_table.sort_index()  # Sort zones 1-263
pivot_table = pivot_table[sorted(pivot_table.columns)]  # Sort columns chronologically

# 9. Save to CSV
# pivot_table.to_csv('hourly_pickups_Feb1-28_2025.csv')
pivot_table.to_csv('hourly_dropoffs_Feb1-28_2025.csv')

# Optional: Display summary
print(f"Result shape: {pivot_table.shape} (zones × hours)")
print(f"Total zones: {len(pivot_table)}")
print(f"Total time points: {len(pivot_table.columns)}")
print("\nSample of pivot table:")
print(pivot_table.head())