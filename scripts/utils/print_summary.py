# scenes=(
     # "7b6477cb95" "c50d2d1d42" "cc5237fd77" "acd95847c5"
     # "31a2c91c43" "e7af285f7d" "286b55a2bf" "7bc286c1b6"
# )
import pandas as pd
import os
import json
from os.path import join
# pd.options.display.float_format = '{:.4f}'.format

benchmark_dir = '/mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark'
output_path_pattern = join(benchmark_dir, 'summary_{}.csv')
output_file_path = join(benchmark_dir, 'summary.csv')
metrics = ['F-score', 'Acc', 'Comp', 'Prec', 'Recall']
scenes = ['7b6477cb95', 'c50d2d1d42', 'cc5237fd77', 'acd95847c5', '31a2c91c43', 'e7af285f7d', '286b55a2bf', '7bc286c1b6']
methods = ['depth_anything_v2', 'marigold', 'lowres_lidar', 'lowres_lidar_bilinear_upsample', 'may_depthanythingmetric_arkitscenes_hypersim_minmax']
align_methods = ['gt']

output_data = {metric: {} for metric in metrics}

for scene in scenes:
    for method in methods:
        for align_method in align_methods:
            data_path = join(benchmark_dir, f'{scene}_{method}/output_align-{align_method}/metrics.json')
            if not os.path.exists(data_path):
                continue
            data_info = json.load(open(data_path))
            
            for metric in metrics:
                if scene not in output_data[metric]:
                    output_data[metric][scene] = {}
                output_data[metric][scene][f'{method}_{align_method}'] = data_info[metric]
     
with open(output_file_path, 'w') as file:
    for metric in metrics:
        data_list = []
        for method in methods:
            for align_method in align_methods:
                method_data = {scene: output_data[metric][scene].get(f'{method}_{align_method}', None) for scene in scenes}
                valid_scores = [score for score in method_data.values() if score is not None]
                average_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
                row_data = {
                    'Method': f'{method}_align-{align_method}',
                    'Average Result': f'{average_score:.4f}' if average_score is not None else 'N/A'
                }
                row_data.update({scene: f'{score:.4f}' if isinstance(score, float) else 'N/A' for scene, score in method_data.items()})
                data_list.append(row_data)

        # 创建DataFrame
        df_metric = pd.DataFrame(data_list)

        # 计算每列的最大宽度
        max_widths = {column: max([len(str(x)) for x in df_metric[column]] + [len(column)]) for column in df_metric.columns}

        # 写入指标名称作为标题
        file.write(f"Metric: {metric}\n")
        
        # 写入标题行
        header = "".join(column.ljust(max_widths[column] + 2) for column in df_metric.columns)
        file.write(header + "\n")
        
        # 写入数据行
        for index, row in df_metric.iterrows():
            row_formatted = "".join(str(row[column]).ljust(max_widths[column] + 2) for column in df_metric.columns)
            file.write(row_formatted + "\n")
        
        # 在表格间添加一个空行作为分隔
        file.write("\n")

    print(f'All metric tables saved to {output_file_path}')           
                
# for metric in metrics:
#     data_list = []
#     for method in methods:
#         for align_method in align_methods:
#             # 汇总该方法的所有场景结果
#             method_data = {scene: output_data[metric][scene][f'{method}_{align_method}'] for scene in scenes}
#             valid_scores = [score for score in method_data.values() if score is not None]
#             average_score = '{:.6f}'.format(sum(valid_scores) / len(valid_scores)) if valid_scores else None
#             # 创建行数据
#             row_data = {
#                 'Method': f'{method}_{align_method}',
#                 'Average Result': average_score
#             }
#             row_data.update({scene: f'{score:.6f}' for scene, score in method_data.items()})
#             data_list.append(row_data)

#     # 创建DataFrame
#     df_metric = pd.DataFrame(data_list)
    
#      # 计算每列的最大宽度
#     max_widths = {column: max([len(str(x)) for x in df_metric[column]] + [len(column)]) for column in df_metric.columns}

#     # 格式化每行数据，确保对齐
#     with open(output_path_pattern.format(metric), 'w') as f:
#         # 写入标题
#         header = "".join(column.ljust(max_widths[column] + 2) for column in df_metric.columns)
#         f.write(header + "\n")
#         # 写入数据行
#         for index, row in df_metric.iterrows():
#             row_formatted = "".join(str(row[column]).ljust(max_widths[column] + 2) for column in df_metric.columns)
#             f.write(row_formatted + "\n")

#     print(f'Aligned text file for {metric} saved to {output_path_pattern.format(metric)}')
    
            





