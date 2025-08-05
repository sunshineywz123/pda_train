import tyro
import os 
from os.path import join
from tqdm.auto import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import json

def compare_metrics(
    json1: str = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/txts/scannetpp/sep13_baseline_post.json',
    json2: str = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/txts/scannetpp/lidar_post.json',
    ) -> None:
    """Test entry1"""
    metric1 = {}
    for line in open(json1).readlines()[1:-1]:
        line_dict = json.loads(','.join(line.split(',')[:-1]))
        metric1.update(line_dict)
    metric2 = {}
    for line in open(json2).readlines()[1:-1]:
        line_dict = json.loads(','.join(line.split(',')[:-1]))
        metric2.update(line_dict)
    
    scenes, diffs, metrics = [], [], [] 
    for k in metric1:
        scenes.append(k)
        diffs.append(metric1[k]['F-score'] - metric2[k]['F-score'])
        metrics.append([metric1[k], metric2[k]])
    sort_ids = np.argsort(diffs)[::-1]
    
    select_scenes = [
    "09c1414f1b", 
    "1ada7a0617",
    "40aec5fffa",
    "3e8bba0176", 
    "acd95847c5",
    "578511c8a9",
    "5f99900f09",
    "c4c04e6d6c",
    "f3d64c30f8",
    "7bc286c1b6",
    "c5439f4607",
    "286b55a2bf",
    "fb5a96b1a2"] # random select
    for idx, sort_id in enumerate(sort_ids):
        if len(select_scenes) >= 20: 
            continue
        if scenes[sort_id] not in select_scenes:
            select_scenes.append(scenes[sort_id])
        print(scenes[sort_id], diffs[sort_id], metrics[sort_id][0]['F-score'], metrics[sort_id][1]['F-score'])
    print(select_scenes)
    
    metric1_list, metric2_list = [], [] 
    for scene in select_scenes:
        print(scene, metric1[scene]['F-score'], metric2[scene]['F-score'])
        metric1_list.append(metric1[scene]['F-score'])
        metric2_list.append(metric2[scene]['F-score'])
    print('mean', np.mean(metric1_list), np.mean(metric2_list))
    # print(np.asarray(scenes)[sort_ids].tolist())
    # print(np.asarray(diffs)[sort_ids].tolist())
    # print(np.asarray(metrics)[sort_ids].tolist())
        
    
def entry2() -> None:
    """Test entry2"""
    pass
    
if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "compare_metrics": compare_metrics,
            "entry2": entry2,
        }
    )

