import geopandas as gpd

def geojson_to_shp(geojson_path, shp_path):
    """
    将GeoJSON文件转换为Shapefile格式
    
    参数:
        geojson_path (str): 输入的GeoJSON文件路径
        shp_path (str): 输出的Shapefile文件路径(不需要扩展名)
    """
    try:
        # 读取GeoJSON文件
        gdf = gpd.read_file(geojson_path)
        
        # 检查坐标系，如果没有则设置WGS84
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        
        # 写入Shapefile
        gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
        
        print(f"转换成功！Shapefile已保存到: {shp_path}.shp")
        
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")

# 示例用法
if __name__ == "__main__":
    input_geojson = "input.geojson"  # 替换为你的GeoJSON文件路径
    output_shp = "output_shapefile"   # 输出的Shapefile路径(不需要.shp扩展名)
    
    geojson_to_shp(input_geojson, output_shp)