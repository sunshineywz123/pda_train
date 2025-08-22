import json
import re
import sys


def replace_depth_path(obj):
    if isinstance(obj, dict):
        return {k: replace_depth_path(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_depth_path(item) for item in obj]
    elif isinstance(obj, str):
        # 替换路径
        return re.sub(
            r'/iag_ad_01/ad/yuanweizhong/datasets/shift/([^/]+-[^/]+)/([^/]+_depth_front\.png)',
            r'/iag_ad_01/ad/yuanweizhong/datasets/shift_depth/\1/\2',
            obj
        )
    else:
        return obj

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python replace_shift_depth_path.py <json文件路径>")
        sys.exit(1)
    json_path = sys.argv[1]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    new_data = replace_depth_path(data)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print("替换完成！")