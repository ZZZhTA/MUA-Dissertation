import pandas as pd

# 加载Parquet文件
df1 = pd.read_parquet('yellow_tripdata_2025-02.parquet')
df2 = pd.read_parquet('green_tripdata_2025-02.parquet')

# with pd.option_context('display.max_columns', None, 
#                        'display.max_rows', None,
#                        'display.width', None,
#                        'display.max_colwidth', None):
#     print(df1.head()) 
# with pd.option_context('display.max_columns', None, 
#                        'display.max_rows', None,
#                        'display.width', None,
#                        'display.max_colwidth', None):
#     print(df2.head())
# 
print(df2) 

# df2.to_csv('green_tripdata_2025-02.csv')

df1.head().to_csv('yellow_tripdata_2025-02.csv')
