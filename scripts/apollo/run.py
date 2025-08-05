import tyro
import os 
from os.path import join
from tqdm.auto import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import json

import sys
sys.path.append('.')

from lib.utils.pylogger import Log

def generate_splits(
    data_dir: str = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/datasets/apollo',
    output_file: str = 'data/pl_htcode/processed_datasets/apollo/train_split.json',
    ) -> None:
    """Test entry1"""
    depth_dir = join(data_dir, 'Depth')
    rgb_dir = join(data_dir, 'RGB')
    
    times = os.listdir(rgb_dir)
    assert len(os.listdir(depth_dir)) == len(times)
    
    rgb_files = []
    depth_files = []
    
    for time in times:
        types = os.listdir(join(rgb_dir, time))
        assert len(types) == len(os.listdir(join(depth_dir, time)))
        
        for type in types:
            deg_types = os.listdir(join(rgb_dir, time, type))
            assert len(deg_types) == len(os.listdir(join(depth_dir, time, type)))
            
            for deg_type in tqdm(deg_types):
                pde_types = os.listdir(join(rgb_dir, time, type, deg_type))
                assert len(pde_types) == len(os.listdir(join(depth_dir, time, type, deg_type)))
                
                for pde_type in pde_types:
                    tra_types = os.listdir(join(rgb_dir, time, type, deg_type, pde_type))
                    assert len(tra_types) == len(os.listdir(join(depth_dir, time, type, deg_type, pde_type)))
                    
                    for tra_type in tra_types:
                        road_types = os.listdir(join(rgb_dir, time, type, deg_type, pde_type, tra_type))
                        assert len(road_types) == len(os.listdir(join(depth_dir, time, type, deg_type, pde_type, tra_type)))
                        
                        for road_type in road_types:
                            last_types = os.listdir(join(rgb_dir, time, type, deg_type, pde_type, tra_type, road_type))
                            assert len(last_types) == len(os.listdir(join(depth_dir, time, type, deg_type, pde_type, tra_type, road_type)))
                            
                            for last_type in last_types:
                                rgb_files_ = os.listdir(join(rgb_dir, time, type, deg_type, pde_type, tra_type, road_type, last_type))
                                depth_files_ = os.listdir(join(depth_dir, time, type, deg_type, pde_type, tra_type, road_type, last_type))
                                
                                rgb_files.extend([join(rgb_dir, time, type, deg_type, pde_type, tra_type, road_type, last_type, file) for file in rgb_files_ if file.replace('.jpg', '.png') in depth_files_])
                                depth_files.extend([join(depth_dir, time, type, deg_type, pde_type, tra_type, road_type, last_type, file) for file in depth_files_ if file.replace('.png', '.jpg') in rgb_files_])
                                
                                if len(rgb_files) != len(depth_files):
                                    import ipdb; ipdb.set_trace()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)                             
    Log.info('Num of rgb_files: {}'.format(len(rgb_files)))
    Log.info('Saving to {}'.format(output_file))
    json.dump({"rgb_files": rgb_files, "depth_files": depth_files}, open(output_file, 'w'))
    
    
    
def read_depth(
    path: str = 'data/pl_htcode/processed_datasets/apollo/train_split.json',
    ) -> None:
    """Test entry2"""
    depth_file = json.load(open(path))['depth_files'][0]    
    bgr = np.asarray(cv2.imread(depth_file)).astype(np.float32) / 255.
    depth = ((bgr[..., 2] + bgr[..., 1] / 255.0) * 65536) / 100.
    import ipdb; ipdb.set_trace()
    
    pass
    
if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "generate_splits": generate_splits,
            "read_depth": read_depth,
        }
    )

