#!/bin/bash
source /root/miniconda3/bin/activate /iag_ad_01/ad/yuanweizhong/miniconda/promptda
export workspace=data/pl_htcode
export exp=aug_shift_comp_baseline_new_lidar_precomp_fov_sparse_lidar_minmax_min
python3 main.py exp=depth_estimation/${exp} +data.train_loader_opts.num_workers=0

# python3 -m ptvsd --host 0.0.0.0 --port 5691 main.py exp=depth_estimation/${exp} +data.train_loader_opts.num_workers=0
# python3 -m ptvsd --host 0.0.0.0 --port 5691 main.py exp=depth_estimation/${exp} entry=debug_cfg +data.train_loader_opts.num_workers=0

# python3 main.py exp=depth_estimation/${exp}
# python3 main.py exp=depth_estimation/${exp} data=depth_estimation/waymo_comp +model.output_tag=waymo_${scene}_756 entry=val exp_name=${exp} +model.near_depth=1. +model.far_depth=50. data.val_dataset.dataset_opts.split_path=data/pl_htcode/processed_datasets/waymo/${scene}.json model.save_orig_pred=True