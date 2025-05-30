import geopandas as gpd
import pandas as pd

# 读取shapefile文件
shp_path = 'd:/Desktop/dissertation/workspace/taxi_zones/taxi_zones.shp'
gdf = gpd.read_file(shp_path)

# 确保坐标系统是WGS84（经纬度坐标系统）
if gdf.crs != 'EPSG:4326':
    gdf = gdf.to_crs('EPSG:4326')

# 计算每个zone的中心点坐标
gdf['longitude'] = gdf.geometry.centroid.x
gdf['latitude'] = gdf.geometry.centroid.y

# 读取现有的taxi_zones.csv文件
taxi_zones_df = pd.read_csv('d:/Desktop/dissertation/workspace/taxi_zones.csv')

# 合并中心点坐标到taxi_zones.csv
taxi_zones_with_coords = taxi_zones_df.merge(
    gdf[['LocationID', 'longitude', 'latitude']],
    on='LocationID',
    how='left'
)

# 保存更新后的taxi_zones.csv
taxi_zones_with_coords.to_csv('d:/Desktop/dissertation/workspace/taxi_zones.csv', index=False)

# 读取od.csv文件
od_df = pd.read_csv('d:/Desktop/dissertation/workspace/od.csv')

# 创建包含LocationID、经纬度和borough的参考数据框
ref_df = taxi_zones_with_coords[['LocationID', 'longitude', 'latitude', 'borough']]

# 为上车地点添加坐标和borough信息
od_with_coords = od_df.merge(
    ref_df,
    left_on='PULocationID',
    right_on='LocationID',
    how='left'
).rename(columns={
    'longitude': 'PU_longitude',
    'latitude': 'PU_latitude',
    'borough': 'PU_borough'
}).drop('LocationID', axis=1)

# 为下车地点添加坐标和borough信息
od_with_coords = od_with_coords.merge(
    ref_df,
    left_on='DOLocationID',
    right_on='LocationID',
    how='left'
).rename(columns={
    'longitude': 'DO_longitude',
    'latitude': 'DO_latitude',
    'borough': 'DO_borough'
}).drop('LocationID', axis=1)

# 保存更新后的od.csv
od_with_coords.to_csv('d:/Desktop/dissertation/workspace/od.csv', index=False)

print("处理完成！")