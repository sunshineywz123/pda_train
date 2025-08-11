import json

train_split_json_file = "data/pl_htcode/processed_datasets/shift/train_split.json"
val_split_json_file = "data/pl_htcode/processed_datasets/shift/val_split.json"

train_split_json = json.load(open(train_split_json_file))
json_data = []
num_data = 5
# 取前5个数据
for i, (rgb_files, depth_files, lowres_files) in enumerate(train_split_json):
    if i >= num_data:  # 只处理前5个
        break
    json_data.append({
        "rgb_files": rgb_files,
        "depth_files": depth_files,
        "lowres_files": lowres_files
    })

json.dump(json_data, open(val_split_json_file, "w"))



