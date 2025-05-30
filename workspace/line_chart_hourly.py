import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import numpy as np

# 加载数据
print("Loading parquet files...")
yellow_df = pd.read_parquet('yellow_tripdata_2025-02.parquet')
green_df = pd.read_parquet('green_tripdata_2025-02.parquet')

# 加载taxi_zones.csv文件以获取zone和borough之间的关系
zones_df = pd.read_csv('taxi_zones.csv')

# 处理Yellow出租车数据
# 转换日期时间列为datetime格式
yellow_df['pickup_datetime'] = pd.to_datetime(yellow_df['tpep_pickup_datetime'])
yellow_df['dropoff_datetime'] = pd.to_datetime(yellow_df['tpep_dropoff_datetime'])

# 提取日期、小时和星期几
yellow_df['pickup_date'] = yellow_df['pickup_datetime'].dt.date
yellow_df['pickup_hour'] = yellow_df['pickup_datetime'].dt.hour
yellow_df['pickup_dayofweek'] = yellow_df['pickup_datetime'].dt.dayofweek

yellow_df['dropoff_date'] = yellow_df['dropoff_datetime'].dt.date
yellow_df['dropoff_hour'] = yellow_df['dropoff_datetime'].dt.hour
yellow_df['dropoff_dayofweek'] = yellow_df['dropoff_datetime'].dt.dayofweek

# 标记工作日和周末 (0-4是工作日，5-6是周末)
yellow_df['pickup_is_weekend'] = yellow_df['pickup_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
yellow_df['dropoff_is_weekend'] = yellow_df['dropoff_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# 合并数据以添加上车地点的borough信息
yellow_pickup = yellow_df.merge(zones_df[['LocationID', 'borough']], 
                    left_on='PULocationID', 
                    right_on='LocationID', 
                    how='left')
yellow_pickup = yellow_pickup.rename(columns={'borough': 'pickup_borough'})

# 合并数据以添加下车地点的borough信息
yellow_dropoff = yellow_df.merge(zones_df[['LocationID', 'borough']], 
                     left_on='DOLocationID', 
                     right_on='LocationID', 
                     how='left')
yellow_dropoff = yellow_dropoff.rename(columns={'borough': 'dropoff_borough'})

# 处理Green出租车数据
# 转换日期时间列为datetime格式
green_df['pickup_datetime'] = pd.to_datetime(green_df['lpep_pickup_datetime'])
green_df['dropoff_datetime'] = pd.to_datetime(green_df['lpep_dropoff_datetime'])

# 提取日期、小时和星期几
green_df['pickup_date'] = green_df['pickup_datetime'].dt.date
green_df['pickup_hour'] = green_df['pickup_datetime'].dt.hour
green_df['pickup_dayofweek'] = green_df['pickup_datetime'].dt.dayofweek

green_df['dropoff_date'] = green_df['dropoff_datetime'].dt.date
green_df['dropoff_hour'] = green_df['dropoff_datetime'].dt.hour
green_df['dropoff_dayofweek'] = green_df['dropoff_datetime'].dt.dayofweek

# 标记工作日和周末 (0-4是工作日，5-6是周末)
green_df['pickup_is_weekend'] = green_df['pickup_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
green_df['dropoff_is_weekend'] = green_df['dropoff_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# 合并数据以添加上车地点的borough信息
green_pickup = green_df.merge(zones_df[['LocationID', 'borough']], 
                    left_on='PULocationID', 
                    right_on='LocationID', 
                    how='left')
green_pickup = green_pickup.rename(columns={'borough': 'pickup_borough'})

# 合并数据以添加下车地点的borough信息
green_dropoff = green_df.merge(zones_df[['LocationID', 'borough']], 
                     left_on='DOLocationID', 
                     right_on='LocationID', 
                     how='left')
green_dropoff = green_dropoff.rename(columns={'borough': 'dropoff_borough'})

# 按小时和borough统计Yellow出租车上车数量，并区分工作日和周末
yellow_pickup_counts = yellow_pickup.groupby(['pickup_hour', 'pickup_borough', 'pickup_is_weekend']).size().reset_index(name='pickup_count')

# 按小时和borough统计Yellow出租车下车数量，并区分工作日和周末
yellow_dropoff_counts = yellow_dropoff.groupby(['dropoff_hour', 'dropoff_borough', 'dropoff_is_weekend']).size().reset_index(name='dropoff_count')

# 按小时和borough统计Green出租车上车数量，并区分工作日和周末
green_pickup_counts = green_pickup.groupby(['pickup_hour', 'pickup_borough', 'pickup_is_weekend']).size().reset_index(name='pickup_count')

# 按小时和borough统计Green出租车下车数量，并区分工作日和周末
green_dropoff_counts = green_dropoff.groupby(['dropoff_hour', 'dropoff_borough', 'dropoff_is_weekend']).size().reset_index(name='dropoff_count')

# 合并Yellow和Green出租车数据
# 对于上车
pickup_counts = pd.concat([yellow_pickup_counts, green_pickup_counts])
pickup_counts = pickup_counts.groupby(['pickup_hour', 'pickup_borough', 'pickup_is_weekend']).sum().reset_index()

# 对于下车
dropoff_counts = pd.concat([yellow_dropoff_counts, green_dropoff_counts])
dropoff_counts = dropoff_counts.groupby(['dropoff_hour', 'dropoff_borough', 'dropoff_is_weekend']).sum().reset_index()

# 计算工作日的平均值
# 首先计算2月份的工作日和周末天数
feb_2025_weekdays = 20  # 2025年2月有20个工作日
feb_2025_weekend_days = 8  # 2025年2月有8个周末日

# 对上车数据进行平均值计算
pickup_counts.loc[pickup_counts['pickup_is_weekend'] == 0, 'pickup_count'] = pickup_counts.loc[pickup_counts['pickup_is_weekend'] == 0, 'pickup_count'] / feb_2025_weekdays
pickup_counts.loc[pickup_counts['pickup_is_weekend'] == 1, 'pickup_count'] = pickup_counts.loc[pickup_counts['pickup_is_weekend'] == 1, 'pickup_count'] / feb_2025_weekend_days

# 对下车数据进行平均值计算
dropoff_counts.loc[dropoff_counts['dropoff_is_weekend'] == 0, 'dropoff_count'] = dropoff_counts.loc[dropoff_counts['dropoff_is_weekend'] == 0, 'dropoff_count'] / feb_2025_weekdays
dropoff_counts.loc[dropoff_counts['dropoff_is_weekend'] == 1, 'dropoff_count'] = dropoff_counts.loc[dropoff_counts['dropoff_is_weekend'] == 1, 'dropoff_count'] / feb_2025_weekend_days

# 创建小时列表 (0-23)
hour_list = list(range(24))

# 获取所有boroughs列表
boroughs = zones_df['borough'].unique()

# 创建目录保存图表
output_dir = 'charts_hourly'
os.makedirs(output_dir, exist_ok=True)

# 设置图表样式
plt.style.use('ggplot')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 增加所有图表的字体大小
plt.rcParams.update({'font.size': 16})

# 为每个borough创建图表，显示工作日和周末的上车和下车数量
for i, borough in enumerate(boroughs):
    plt.figure(figsize=(14, 8))
    
    # 获取该borough的工作日上车数据
    weekday_pickup_data = pickup_counts[(pickup_counts['pickup_borough'] == borough) & 
                                       (pickup_counts['pickup_is_weekend'] == 0)]
    full_weekday_pickup_data = pd.DataFrame({'pickup_hour': hour_list})
    full_weekday_pickup_data = full_weekday_pickup_data.merge(weekday_pickup_data, 
                                                            on='pickup_hour', 
                                                            how='left').fillna(0)
    
    # 获取该borough的工作日下车数据
    weekday_dropoff_data = dropoff_counts[(dropoff_counts['dropoff_borough'] == borough) & 
                                         (dropoff_counts['dropoff_is_weekend'] == 0)]
    full_weekday_dropoff_data = pd.DataFrame({'dropoff_hour': hour_list})
    full_weekday_dropoff_data = full_weekday_dropoff_data.merge(weekday_dropoff_data, 
                                                              on='dropoff_hour', 
                                                              how='left').fillna(0)
    
    # 获取该borough的周末上车数据
    weekend_pickup_data = pickup_counts[(pickup_counts['pickup_borough'] == borough) & 
                                       (pickup_counts['pickup_is_weekend'] == 1)]
    full_weekend_pickup_data = pd.DataFrame({'pickup_hour': hour_list})
    full_weekend_pickup_data = full_weekend_pickup_data.merge(weekend_pickup_data, 
                                                            on='pickup_hour', 
                                                            how='left').fillna(0)
    
    # 获取该borough的周末下车数据
    weekend_dropoff_data = dropoff_counts[(dropoff_counts['dropoff_borough'] == borough) & 
                                         (dropoff_counts['dropoff_is_weekend'] == 1)]
    full_weekend_dropoff_data = pd.DataFrame({'dropoff_hour': hour_list})
    full_weekend_dropoff_data = full_weekend_dropoff_data.merge(weekend_dropoff_data, 
                                                              on='dropoff_hour', 
                                                              how='left').fillna(0)
    
    # 绘制工作日上车折线
    plt.plot(full_weekday_pickup_data['pickup_hour'], 
             full_weekday_pickup_data['pickup_count'], 
             label='Weekday Pickups', 
             color=colors[0], 
             marker='o', 
             markersize=6, 
             linewidth=2)
    
    # 绘制工作日下车折线
    plt.plot(full_weekday_dropoff_data['dropoff_hour'], 
             full_weekday_dropoff_data['dropoff_count'], 
             label='Weekday Dropoffs', 
             color=colors[1], 
             marker='o', 
             markersize=6, 
             linewidth=2)
    
    # 绘制周末上车折线
    plt.plot(full_weekend_pickup_data['pickup_hour'], 
             full_weekend_pickup_data['pickup_count'], 
             label='Weekend Pickups', 
             color=colors[2], 
             marker='o', 
             markersize=6, 
             linewidth=2)
    
    # 绘制周末下车折线
    plt.plot(full_weekend_dropoff_data['dropoff_hour'], 
             full_weekend_dropoff_data['dropoff_count'], 
             label='Weekend Dropoffs', 
             color=colors[3], 
             marker='o', 
             markersize=6, 
             linewidth=2)
    
    # 设置标签和标题
    plt.title(f'Hourly Average Pickup and Dropoff Counts for {borough} Borough', fontsize=18)
    plt.xlabel('Hour of Day', fontsize=16)
    plt.ylabel('Average Count', fontsize=16)
    
    # 添加图例
    plt.legend(fontsize=14)
    
    # 启用网格
    plt.grid(True)
    
    # 设置x轴范围和刻度
    plt.xlim([0, 23])
    plt.xticks(range(0, 24, 2))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'{output_dir}/{borough}_hourly_pickups_dropoffs.png')
    plt.close()

print("All hourly charts have been generated and saved to the 'charts_hourly' directory")