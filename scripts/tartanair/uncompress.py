import os
from os.path import join
import sys
import shutil
sys.path.append('.')
from lib.utils.parallel_utils import parallel_execution

def create_dir(directory):
    """创建单个目录"""
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"在创建目录时发生错误: {e}")

def copy_file(input_tuple):
    """复制单个文件到目标目录"""
    try:
        source, destination = input_tuple
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        print(f"在复制文件时发生错误: {e}")
        return False


def process_one_scene(scene, easy_type, data_type):
    cmd = 'ossutil64 cp oss://antsys-vilab/xiulin/ht/data/{}/{}/{}.zip /root/'.format(scene, easy_type, data_type)
    os.system(cmd)
    
    workspace = '/root/data'
    os.system('mkdir -p {}'.format(workspace))
    os.system('unzip -o /root/{}.zip -d {}'.format(data_type, workspace))
    
    dirs_to_create = []
    files_to_copy = []
    source_folder = join(workspace, '{}/{}'.format(scene, scene))
    destination_folder = join('/input/datasets/TartanAir/train', '{}'.format(scene))
    for root, dirs, files in os.walk(source_folder):
        for dir in dirs:
            dirs_to_create.append(os.path.join(root, dir).replace(source_folder, destination_folder, 1))
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = src_file.replace(source_folder, destination_folder, 1)
            files_to_copy.append((src_file, dst_file))
    parallel_execution(dirs_to_create, action=create_dir, num_processes=16, print_progress=True)
    parallel_execution(files_to_copy, action=copy_file, num_processes=16, print_progress=True)
    
    os.system('rm -rf {}'.format(workspace))
    os.system('rm -rf /root/{}.zip'.format(data_type))

if __name__ == '__main__':
    # scenes = ['abandonedfactory']
    scenes = ['abandonedfactory', 'abandonedfactory_night', 'amusement', 'carwelding', 'endofworld', 'gascola', 'hospital', 'japanesealley', 'neighborhood', 'ocean', 'office', 'office2', 'oldtown', 'seasidetown', 'seasonsforest', 'seasonsforest_winter', 'soulcity', 'westerndesert']
    for scene in scenes:
        for easy_type in ['Easy', 'Hard']:
            for data_type in ['depth_left', 'depth_right', 'rgb_left', 'rgb_right']:
                process_one_scene(scene, easy_type, data_type)