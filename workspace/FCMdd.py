import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tslearn.metrics import dtw_path
import os

# Set random seed for reproducibility
np.random.seed(42)

# ========================
# 1. Data Loading and Preprocessing
# ========================

def load_taxi_data(pickups_file, dropoffs_file, first_n_days=14, exclude_zones=[1, 132, 138, 70]):
    """
    Load NYC taxi pickup and dropoff data
    
    Parameters:
        pickups_file: Path to pickup data file
        dropoffs_file: Path to dropoff data file
        first_n_days: Number of days to include (default: 14)
        exclude_zones: List of zone IDs to exclude from analysis
        
    Returns:
        X: Time series data with shape (n_zones, n_timepoints, n_features=2)
        zone_ids: List of zone IDs
    """
    # Read pickup and dropoff data
    pickups_df = pd.read_csv(pickups_file)
    dropoffs_df = pd.read_csv(dropoffs_file)
    
    # Filter out excluded zones
    mask = ~pickups_df['PULocationID'].isin(exclude_zones)
    pickups_df = pickups_df[mask]
    dropoffs_df = dropoffs_df[mask]
    
    # Extract zone IDs
    zone_ids = pickups_df['PULocationID'].values
    
    # Extract time series data (excluding zone ID column)
    hours_per_day = 24
    total_hours = first_n_days * hours_per_day
    
    # Select only the first n days of data
    pickups_data = pickups_df.iloc[:, 1:total_hours+1].values  # +1 because we exclude the first column
    dropoffs_data = dropoffs_df.iloc[:, 1:total_hours+1].values
    
    # Check data shape
    n_zones = len(zone_ids)
    n_timepoints = pickups_data.shape[1]  # Should be first_n_days * 24
    
    print(f"Loaded {n_zones} zones, each with {n_timepoints} timepoints (first {first_n_days} days)")
    
    # Create 3D array: (n_zones, n_timepoints, 2)
    X = np.zeros((n_zones, n_timepoints, 2))
    X[:, :, 0] = pickups_data  # First feature: pickup count
    X[:, :, 1] = dropoffs_data  # Second feature: dropoff count
    
    return X, zone_ids

def normalize_data(X):
    """
    Normalize time series data
    
    Parameters:
        X: Original time series data with shape (n_samples, n_timepoints, n_features)
        
    Returns:
        X_norm: Normalized data with same shape as X
    """
    n_samples, n_timepoints, n_features = X.shape
    X_reshaped = X.reshape(n_samples, -1)  # Combine time and feature dimensions
    
    # Use StandardScaler for normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    
    # Restore original shape
    X_norm = X_scaled.reshape(n_samples, n_timepoints, n_features)
    
    return X_norm

# ========================
# 2. DTW Distance Calculation (using tslearn)
# ========================

def dtw_distance(x, y):
    """
    Calculate DTW distance between two multivariate time series using tslearn

    Parameters:
        x: First time series with shape (T1, D)
        y: Second time series with shape (T2, D)

    Returns:
        float: DTW distance
    """
    _, cost = dtw_path(x, y)
    return cost

# ========================
# 3. Build DTW Distance Matrix
# ========================

def build_dtw_matrix(X):
    """
    Build DTW distance matrix for all samples

    Parameters:
        X: Time series data with shape (n_samples, T, D)

    Returns:
        distance_matrix: Matrix with shape (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    dtw_matrix = np.zeros((n_samples, n_samples))

    for i in tqdm(range(n_samples), desc="Building DTW distance matrix"):
        for j in range(i, n_samples):
            dist = dtw_distance(X[i], X[j])
            dtw_matrix[i, j] = dist
            dtw_matrix[j, i] = dist  # Symmetrize

    return dtw_matrix

# ========================
# 4. Fuzzy C-medoids Clustering Algorithm
# ========================

def fuzzy_c_medoids_with_dtw(dtw_matrix, c, m=2, max_iter=100, epsilon=1e-4):
    """
    Run fuzzy C-medoids clustering using precomputed DTW distance matrix

    Parameters:
        dtw_matrix: Precomputed distance matrix with shape (n_samples, n_samples)
        c: Number of clusters
        m: Fuzziness exponent (default: 2)
        max_iter: Maximum number of iterations
        epsilon: Convergence threshold

    Returns:
        U: Fuzzy membership matrix with shape (n_samples, c)
        V: List of medoid indices
    """
    n_samples = dtw_matrix.shape[0]
    U = np.random.rand(n_samples, c)
    U = U / U.sum(axis=1, keepdims=True)  # Normalize memberships

    # Randomly select initial medoids
    V = np.random.choice(n_samples, size=c, replace=False)

    for iter in range(max_iter):
        # Update memberships
        U_prev = U.copy()
        for i in range(n_samples):
            for k in range(c):
                d_ik = dtw_matrix[i, V[k]]  # Get DTW distance from matrix
                
                # Handle special case: if sample is the medoid itself
                if d_ik == 0:
                    # Set membership to 1 for this cluster, 0 for others
                    U[i, :] = 0
                    U[i, k] = 1
                    break
                    
                # Calculate membership
                denominator = 0
                for j in range(c):
                    d_ij = dtw_matrix[i, V[j]]
                    if d_ij == 0:  # Avoid division by zero
                        denominator = np.inf
                        break
                    denominator += (d_ik / d_ij) ** (2 / (m - 1))
                    
                if denominator > 0:
                    U[i, k] = 1 / denominator
                else:
                    print(f"Warning: Sample {i} has problematic membership for cluster {k}, denominator is {denominator}")
                    # Set a default value
                    U[i, k] = 0

        # Update medoids
        V_new = []
        for k in range(c):
            min_cost = float('inf')
            best_medoid = V[k]
            for candidate in range(n_samples):
                # Calculate objective function value for candidate medoid
                cost = sum(
                    U[i, k] ** m * dtw_matrix[i, candidate] ** 2
                    for i in range(n_samples)
                )
                if cost < min_cost:
                    min_cost = cost
                    best_medoid = candidate
            V_new.append(best_medoid)

        # Check convergence
        if np.linalg.norm(np.array(V) - np.array(V_new)) < epsilon:
            break
        V = V_new

    return U, V

# ========================
# 5. Cluster Evaluation Metrics
# ========================

def calculate_sse(X, U, V, dtw_matrix, m=2):
    """
    Calculate Sum of Squared Errors (SSE) for fuzzy clustering
    
    Parameters:
        X: Time series data with shape (n_samples, T, D)
        U: Membership matrix with shape (n_samples, c)
        V: List of medoid indices
        dtw_matrix: Precomputed DTW distance matrix
        m: Fuzziness exponent
        
    Returns:
        float: SSE value
    """
    n_samples = X.shape[0]
    c = len(V)
    sse = 0.0
    
    for i in range(n_samples):
        for k in range(c):
            # Get DTW distance from precomputed matrix instead of recalculating
            dist = dtw_matrix[i, V[k]]
            
            # Check if distance is valid
            if np.isnan(dist) or np.isinf(dist):
                print(f"Warning: Sample {i} has distance {dist} to cluster center {k}")
                continue
                
            # Check if membership is valid
            if np.isnan(U[i, k]) or np.isinf(U[i, k]):
                print(f"Warning: Sample {i} has membership {U[i, k]} for cluster {k}")
                continue
                
            contribution = U[i, k]**m * dist**2
            
            # Check if contribution is valid
            if not np.isnan(contribution) and not np.isinf(contribution):
                sse += contribution
            else:
                print(f"Warning: Sample {i} has SSE contribution {contribution} for cluster {k}")
    
    return sse

def calculate_silhouette(X, U, V, dtw_matrix):
    """
    Calculate silhouette coefficient for fuzzy clustering
    
    Parameters:
        X: Time series data with shape (n_samples, T, D)
        U: Membership matrix with shape (n_samples, c)
        V: List of medoid indices
        dtw_matrix: Precomputed DTW distance matrix
        
    Returns:
        float: Silhouette coefficient
    """
    # Convert fuzzy clustering to hard clustering (take max membership)
    labels = np.argmax(U, axis=1)
    
    # Use the precomputed DTW matrix directly
    silhouette = silhouette_score(dtw_matrix, labels, metric='precomputed')
    return silhouette

# ========================
# 6. Find Optimal Number of Clusters
# ========================

def find_optimal_clusters(X, dtw_matrix, max_clusters=10):
    """
    Find optimal number of clusters using silhouette coefficient and SSE
    
    Parameters:
        X: Time series data with shape (n_samples, T, D)
        dtw_matrix: Precomputed DTW distance matrix
        max_clusters: Maximum number of clusters to evaluate
        
    Returns:
        int: Optimal number of clusters
    """
    silhouette_scores = []
    sse_values = []
    
    # Start evaluation from 2 clusters
    for c in range(2, max_clusters + 1):
        print(f"Evaluating {c} clusters...")
        U, V = fuzzy_c_medoids_with_dtw(dtw_matrix, c=c)
        
        # Calculate silhouette coefficient
        print("Calculating silhouette...")
        silhouette = calculate_silhouette(X, U, V, dtw_matrix)
        silhouette_scores.append(silhouette)
        
        # Calculate SSE
        print("Calculating SSE...")
        sse = calculate_sse(X, U, V, dtw_matrix)
        sse_values.append(sse)
        
        print(f"Clusters {c}: Silhouette = {silhouette:.4f}, SSE = {sse:.4f}")
    
    # Plot silhouette coefficient and SSE curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient vs. Number of Clusters')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), sse_values, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('SSE vs. Number of Clusters')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cluster_evaluation.png')
    plt.show()
    
    # Determine optimal number of clusters based on silhouette coefficient
    optimal_c = np.argmax(silhouette_scores) + 2  # +2 because we start from 2
    print(f"Based on silhouette coefficient, optimal number of clusters: {optimal_c}")
    
    return optimal_c

# ========================
# 7. Visualize Clustering Results
# ========================


def visualize_weekday_weekend_patterns(X, V, zone_ids):
    """
    Visualize weekday vs weekend patterns for each medoid zone
    
    Parameters:
        X: Time series data with shape (n_samples, T, D)
        V: List of medoid indices
        zone_ids: List of zone IDs
    """
    n_clusters = len(V)
    hours_per_day = 24
    days = X.shape[1] // hours_per_day
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 5 * n_clusters))
    if n_clusters == 1:
        axes = [axes]  # Ensure axes is a list even with only one subplot
    
    # Define weekday and weekend indices
    # Assuming data starts from day 1 (Feb 1, 2025) which is a Saturday
    # So weekends are days 0-1, 7-8, etc. (0-indexed)
    weekend_indices = []
    weekday_indices = []
    
    for day in range(days):
        # Feb 1, 2025 is a Saturday (day 5 of week where Monday is 0)
        # So day 0 is Saturday, day 1 is Sunday, etc.
        day_of_week = (day + 5) % 7  # 5, 6 are weekend (Sat, Sun)
        
        day_start = day * hours_per_day
        day_end = (day + 1) * hours_per_day
        
        if day_of_week >= 5:  # Weekend (Saturday or Sunday)
            weekend_indices.extend(range(day_start, day_end))
        else:  # Weekday
            weekday_indices.extend(range(day_start, day_end))
    
    # Process each medoid
    for k, medoid_idx in enumerate(V):
        ax = axes[k]
        zone_id = zone_ids[medoid_idx]
        
        # Extract data for this medoid
        pickup_data = X[medoid_idx, :, 0]
        dropoff_data = X[medoid_idx, :, 1]
        
        # Calculate average hourly patterns
        weekday_pickup_hourly = np.zeros(hours_per_day)
        weekday_dropoff_hourly = np.zeros(hours_per_day)
        weekend_pickup_hourly = np.zeros(hours_per_day)
        weekend_dropoff_hourly = np.zeros(hours_per_day)
        
        # Calculate averages for weekdays
        for hour in range(hours_per_day):
            weekday_hours = [i for i in weekday_indices if i % hours_per_day == hour]
            if weekday_hours:
                weekday_pickup_hourly[hour] = np.mean(pickup_data[weekday_hours])
                weekday_dropoff_hourly[hour] = np.mean(dropoff_data[weekday_hours])
        
        # Calculate averages for weekends
        for hour in range(hours_per_day):
            weekend_hours = [i for i in weekend_indices if i % hours_per_day == hour]
            if weekend_hours:
                weekend_pickup_hourly[hour] = np.mean(pickup_data[weekend_hours])
                weekend_dropoff_hourly[hour] = np.mean(dropoff_data[weekend_hours])
        
        # Plot data
        hours = np.arange(hours_per_day)
        
        # Pickups
        ax.plot(hours, weekday_pickup_hourly, 'b-', label='Weekday Pickups', linewidth=2)
        ax.plot(hours, weekend_pickup_hourly, 'b--', label='Weekend Pickups', linewidth=2)
        
        # Dropoffs
        ax.plot(hours, weekday_dropoff_hourly, 'r-', label='Weekday Dropoffs', linewidth=2)
        ax.plot(hours, weekend_dropoff_hourly, 'r--', label='Weekend Dropoffs', linewidth=2)
        
        # Set labels and title
        ax.set_title(f'Weekday vs Weekend Patterns for Zone {zone_id} (Cluster {k+1})')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Trip Count')
        ax.set_xticks(np.arange(0, 24, 2))
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout(pad=3.0)  # Increase padding between subplots
    plt.savefig('weekday_weekend_patterns.png')
    plt.show()

def visualize_cluster_medoids(X, V, zone_ids):
    """
    Visualize medoids (representative zones) for each cluster
    
    Parameters:
        X: Time series data with shape (n_samples, T, D)
        V: List of medoid indices
        zone_ids: List of zone IDs
    """
    n_clusters = len(V)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 4 * n_clusters))
    
    if n_clusters == 1:
        axes = [axes]  # Ensure axes is a list even with only one subplot
    
    for k, medoid_idx in enumerate(V):
        ax = axes[k]
        zone_id = zone_ids[medoid_idx]
        
        # Plot pickup and dropoff counts
        time_points = np.arange(X.shape[1])  # Time points
        ax.plot(time_points, X[medoid_idx, :, 0], 'b-', label='Pickup Count')
        ax.plot(time_points, X[medoid_idx, :, 1], 'r-', label='Dropoff Count')
        
        ax.set_title(f'Representative Zone for Cluster {k+1} (Zone ID: {zone_id})')
        ax.set_xlabel('Time Point (Hour)')
        ax.set_ylabel('Trip Count')  # 修改为Trip Count
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout(pad=3.0)  # 增加子图之间的间距
    plt.savefig('fuzzy_c_medoids_results.png')
    plt.show()

def main():
    # Load data
    pickups_file = "hourly_pickups_Feb1-28_2025.csv"
    dropoffs_file = "hourly_dropoffs_Feb1-28_2025.csv"
    X, zone_ids = load_taxi_data(pickups_file, dropoffs_file, first_n_days=14, exclude_zones=[1, 132, 138, 70])
    
    # Normalize data
    print("Normalizing data...")
    X_norm = normalize_data(X)
    
    # Build or load DTW distance matrix
    dtw_matrix_file = 'dtw_matrix_14days.npy'
    if os.path.exists(dtw_matrix_file):
        print(f"Loading saved DTW distance matrix: {dtw_matrix_file}")
        dtw_matrix = np.load(dtw_matrix_file)
        print(f"DTW distance matrix shape: {dtw_matrix.shape}")
        
        # Check if matrix dimensions match current data
        if dtw_matrix.shape[0] != X_norm.shape[0]:
            print("Warning: Saved DTW matrix dimensions don't match current data.")
            print("Rebuilding DTW distance matrix...")
            dtw_matrix = build_dtw_matrix(X_norm)
            np.save('dtw_matrix_14days.npy', dtw_matrix)
            print("DTW distance matrix saved as dtw_matrix_14days.npy")
    else:
        print("Building DTW distance matrix...")
        dtw_matrix = build_dtw_matrix(X_norm)
        np.save('dtw_matrix_14days.npy', dtw_matrix)
        print("DTW distance matrix saved as dtw_matrix_14days.npy")
    
    # Set number of clusters
    optimal_c = 3
    # Define best cluster number
    # print("Defining best cluster number...")
    # optimal_c = find_optimal_clusters(X_norm, dtw_matrix, max_clusters=8)
    
    # Run fuzzy C-medoids with optimal number of clusters
    print(f"Running fuzzy C-medoids with {optimal_c} clusters...")
    U, V = fuzzy_c_medoids_with_dtw(dtw_matrix, c=optimal_c)
    
    # Display membership for first 10 zones
    print("\nMembership for first 10 zones:")
    print(pd.DataFrame(U[:10], columns=[f"Cluster_{i+1}" for i in range(optimal_c)]))

    # Display final medoid indices and corresponding zone IDs
    print("\nFinal medoid indices and zone IDs:")
    for i, medoid_idx in enumerate(V):
        print(f"Cluster {i+1}: Medoid index = {medoid_idx}, Zone ID = {zone_ids[medoid_idx]}")
    
    # Visualize clustering results
    print("\nVisualizing clustering results...")
    visualize_cluster_medoids(X, V, zone_ids)

    # 添加工作日与周末模式的可视化
    print("\nVisualizing weekday vs weekend patterns...")
    visualize_weekday_weekend_patterns(X, V, zone_ids)

    # 在main函数中，在显示membership信息之后添加以下代码

    # 保存隶属度矩阵到CSV文件
    print("\nSaving fuzzy membership matrix to CSV...")
    membership_df = pd.DataFrame(U, columns=[f"Cluster_{i+1}" for i in range(optimal_c)])
    
    # 添加Medoid index和Zone ID信息
    membership_df['Zone_ID'] = zone_ids
    
    # 添加每个样本对应的Medoid信息
    medoid_info = []
    for i in range(len(U)):
        # 获取最大隶属度对应的簇
        max_cluster_idx = np.argmax(U[i])
        # 获取该簇的Medoid索引
        medoid_idx = V[max_cluster_idx]
        # 获取该Medoid对应的Zone ID
        medoid_zone_id = zone_ids[medoid_idx]
        medoid_info.append({
            'Medoid_Index': medoid_idx,
            'Medoid_Zone_ID': medoid_zone_id
        })
    
    # 将Medoid信息添加到DataFrame
    medoid_info_df = pd.DataFrame(medoid_info)
    membership_df = pd.concat([membership_df, medoid_info_df], axis=1)
    
    # 保存到CSV文件
    csv_filename = f"fuzzy_membership_c{optimal_c}.csv"
    membership_df.to_csv(csv_filename, index=True, index_label='Sample_Index')
    print(f"Fuzzy membership matrix saved to {csv_filename}")

if __name__ == "__main__":
    main()