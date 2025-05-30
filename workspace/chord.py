# 曼哈顿內部行程占比实在太大，弦图画出来不好看
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path
import matplotlib.patches as patches

# 读取数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 按borough聚合数据
def aggregate_by_borough(df):
    # 确保所有borough名称都是有效的
    df = df.dropna(subset=['PU_borough', 'DO_borough'])
    
    # 按起始和目的地borough分组并求和
    borough_flow = df.groupby(['PU_borough', 'DO_borough'])['trip_count'].sum().reset_index()
    
    # 创建一个包含所有borough组合的矩阵
    boroughs = sorted(list(set(borough_flow['PU_borough'].unique()) | set(borough_flow['DO_borough'].unique())))
    flow_matrix = pd.DataFrame(0, index=boroughs, columns=boroughs)
    
    # 填充矩阵
    for _, row in borough_flow.iterrows():
        flow_matrix.loc[row['PU_borough'], row['DO_borough']] = row['trip_count']
    
    return flow_matrix, boroughs

# 绘制平行坐标弦图
def plot_parallel_chord_diagram(matrix, labels):
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置颜色
    n = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    
    # 计算每个borough的总流出和流入量
    outgoing_flow = matrix.sum(axis=1)
    incoming_flow = matrix.sum(axis=0)
    
    # 计算总流量用于归一化
    total_outgoing = outgoing_flow.sum()
    total_incoming = incoming_flow.sum()
    
    # 设置左右两列的位置
    left_x = 0.1
    right_x = 0.9
    
    # 计算每个borough在左右两列中的位置和高度
    left_positions = {}
    right_positions = {}
    
    # 左侧列（出发地）
    current_y = 0.05
    for label in labels:
        height = (outgoing_flow[label] / total_outgoing) * 0.9  # 0.9是可用高度
        left_positions[label] = (current_y, height)
        current_y += height
    
    # 右侧列（目的地）
    current_y = 0.05
    for label in labels:
        height = (incoming_flow[label] / total_incoming) * 0.9  # 0.9是可用高度
        right_positions[label] = (current_y, height)
        current_y += height
    
    # 绘制左侧列（出发地）
    for i, label in enumerate(labels):
        start_y, height = left_positions[label]
        rect = patches.Rectangle((left_x, start_y), 0.05, height, 
                               facecolor=colors[i], alpha=0.8, edgecolor='none')
        ax.add_patch(rect)
        
        # 添加标签
        ax.text(left_x - 0.02, start_y + height/2, label, 
                ha='right', va='center', fontsize=12, fontweight='bold')
    
    # 绘制右侧列（目的地）
    for i, label in enumerate(labels):
        start_y, height = right_positions[label]
        rect = patches.Rectangle((right_x - 0.05, start_y), 0.05, height, 
                               facecolor=colors[i], alpha=0.8, edgecolor='none')
        ax.add_patch(rect)
        
        # 添加标签
        ax.text(right_x + 0.02, start_y + height/2, label, 
                ha='left', va='center', fontsize=12, fontweight='bold')
    
    # 绘制连接弧（表示borough之间的流量）
    for i, source in enumerate(labels):
        for j, target in enumerate(labels):
            if matrix.loc[source, target] > 0:
                # 获取源和目标的位置
                source_start_y, source_height = left_positions[source]
                target_start_y, target_height = right_positions[target]
                
                # 计算流量比例
                flow_value = matrix.loc[source, target]
                source_ratio = flow_value / outgoing_flow[source]
                target_ratio = flow_value / incoming_flow[target]
                
                # 计算连接线的起点和终点
                source_y = source_start_y + source_height * source_ratio / 2
                target_y = target_start_y + target_height * target_ratio / 2
                
                # 设置连接线的宽度和透明度
                width = max(0.5, 5 * flow_value / matrix.values.max())  # 最小宽度为0.5
                alpha = min(0.9, max(0.2, flow_value / matrix.values.max()))  # 透明度范围：0.2-0.9
                
                # 创建贝塞尔曲线路径
                verts = [
                    (left_x + 0.05, source_y),  # 起点
                    ((left_x + right_x - 0.05) / 2, (source_y + target_y) / 2),  # 控制点
                    (right_x - 0.05, target_y)  # 终点
                ]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)
                
                # 绘制路径
                patch = patches.PathPatch(path, facecolor='none', 
                                         edgecolor=colors[i], lw=width, alpha=alpha)
                ax.add_patch(patch)
    
    # 设置图表属性
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title('Borough Flow Diagram', fontsize=16)
    plt.tight_layout()
    plt.savefig('borough_flow_parallel_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主函数
def main():
    # 加载数据
    df = load_data('od.csv')
    
    # 聚合数据
    flow_matrix, boroughs = aggregate_by_borough(df)
    
    # 打印流量矩阵
    print("Borough Flow Matrix:")
    print(flow_matrix)
    
    # 绘制平行坐标弦图
    plot_parallel_chord_diagram(flow_matrix, boroughs)

if __name__ == "__main__":
    main()