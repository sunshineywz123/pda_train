#!/bin/bash
source /root/miniconda3/bin/activate /iag_ad_01/ad/yuanweizhong/miniconda/promptda
export workspace=data/pl_htcode
export exp=aug_shift_comp_baseline_new_lidar_precomp_fov_sparse_lidar_minmax_min
# python3 main.py exp=depth_estimation/${exp} entry=val +data.train_loader_opts.num_workers=0


# python3 main.py exp=depth_estimation/${exp} entry=predict
python3 -m ptvsd --host 0.0.0.0 --port 5691 main.py exp=depth_estimation/${exp} entry=predict +data.val_loader_opts.num_workers=0
# python3 -m ptvsd --host 0.0.0.0 --port 5691 main.py exp=depth_estimation/${exp} entry=debug_val_dataloader +data.train_loader_opts.num_workers=0