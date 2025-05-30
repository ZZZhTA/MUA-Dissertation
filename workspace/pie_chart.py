import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
data = pd.read_csv('tabulate.csv')

# Create a dictionary to map building type codes to their meanings based on the image
building_types = {
    'A_01': 'One & Two Family Buildings',
    'A_02': 'Multi-Family Walk-Up Buildings',
    'A_03': 'Multi-Family Elevator Buildings',
    'A_04': 'Mixed Residential & Commercial Buildings',
    'A_05': 'Commercial & Office Buildings',
    'A_06': 'Industrial & Manufacturing',
    'A_07': 'Transportation & Utility',
    'A_08': 'Public Facilities & Institutions',
    'A_09': 'Open Space & Outdoor Recreation',
    'A_10': 'Parking Facilities',
    'A_11': 'Vacant Land'
}

# Reshape the data for easier processing
reshaped_data = []
for index, row in data.iterrows():
    bd_type = row['bd_type']
    for zone in ['c1', 'c2', 'c3']:
        reshaped_data.append({
            'zone_type': zone,
            'building_type': bd_type,
            'area': row[zone]
        })

reshaped_df = pd.DataFrame(reshaped_data)

# Create pie charts for the three zone types
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Colors for consistency across charts
colors = plt.cm.tab20(np.linspace(0, 1, len(building_types)))

# Process each zone type
for i, zone_type in enumerate(['c1', 'c2', 'c3']):
    ax = axes[i]
    
    # Extract building area data for this zone
    zone_data = reshaped_df[reshaped_df['zone_type'] == zone_type]
    
    # If no data for this zone, display empty pie chart
    if len(zone_data) == 0:
        ax.set_title(f'Zone {zone_type} - No Data')
        ax.pie([1], labels=['No Data'], colors=['lightgray'])
        continue
    
    # Extract building types and areas
    building_types_in_zone = zone_data['building_type'].tolist()
    areas = zone_data['area'].tolist()
    
    # Create pie chart without percentage labels
    ax.pie(areas, colors=colors, startangle=90)
    ax.axis('equal')  # Ensure pie chart is circular
    ax.set_title(f'Building Area Proportion in Zone {zone_type}')

plt.tight_layout()
plt.savefig('building_area_pie_charts.png', dpi=300, bbox_inches='tight')

# Create a separate legend figure
fig_legend = plt.figure(figsize=(10, 6))

# Create patches for the legend
patches = []
for i, (code, label) in enumerate(building_types.items()):
    patch = plt.Rectangle((0, 0), 1, 1, fc=colors[i])
    patched_label = f"{code.split('_')[1]}: {label}"
    patches.append((patch, patched_label))

# Create the legend
fig_legend.legend([p for p, l in patches], [l for p, l in patches], 
                 loc='center', frameon=False, ncol=1)

# Remove axes
plt.axis('off')
plt.tight_layout()
plt.savefig('building_types_legend.png', dpi=300, bbox_inches='tight')

plt.show()

print("Pie charts saved as building_area_pie_charts.png")
print("Legend saved as building_types_legend.png")