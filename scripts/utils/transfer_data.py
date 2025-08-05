import os
from os.path import join
import numpy as np
import json


exp = os.environ['exp']
is_disparity = os.environ['is_disparity'] == 'True'
input_dir = join(os.environ['workspace'], 'outputs/depth_estimation/{}/results'.format(exp))
scene = os.environ['scene']
output_dir = f'/mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark/{scene}_{exp}/depth'
os.makedirs(output_dir, exist_ok=True)
depth_dir = join(input_dir, scene, 'orig_pred')

for root, dirs, files in os.walk(depth_dir):
    for file in files:
        if file.endswith('.npz'):
            src_path = join(root, file)
            tar_path = join(output_dir, file[-10:])
            os.system(f'mv {src_path} {tar_path}')
            
config = {
    'scene_name': scene,
    'is_disparity': is_disparity,
    'is_metric': False,
    'align_methods': ['gt'],
    'method': exp
}
json_output_path = f'/mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark/{scene}_{exp}/config.json'
json.dump(config, open(json_output_path, 'w'))