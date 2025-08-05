# Waymo completion 

## Prepartion

### testing

1. Prepare waymo data following the instructions from StreetGaussian （直到跑完prepare_lidar_depth.py那一步），我使用了019 sequence作为测试场景。
2. 
```
python3 scripts/waymo/generate_splits.py --input $DATA_PATH_TO_WAYMO --seq 019
```

### training 

1. 准备shift数据集
2. 
```
python3 scripts/shift/generate_splits.py --input $IMAGES_PATH_TO_SHIFT --output data/pl_htcode/processed_datasets/shift/train_split.json --tag train
```


### model


1. 预训练模型：data/pl_htcode/cache_models/depth_anything/checkpoints/v2_model_metric_ft_shift_minmax_fov.ckpt

2. 测试模型：data/pl_htcode/outputs/depth_estimation/aug_shift_comp_baseline_new_lidar_precomp_fov_sparse_lidar_minmax_min/checkpoints/e099-s200000.ckpt




## testing 
```
export workspace=data/pl_htcode
export scene=019
export exp=aug_shift_comp_baseline_new_lidar_precomp_fov_sparse_lidar_minmax_min
python3 main.py exp=depth_estimation/${exp} data=depth_estimation/waymo_comp +model.output_tag=waymo_${scene}_756 entry=val exp_name=${exp} +model.near_depth=1. +model.far_depth=50. data.val_dataset.dataset_opts.split_path=data/pl_htcode/processed_datasets/waymo/${scene}.json model.save_orig_pred=True
```

## training 
```
export workspace=data/pl_htcode
export exp=aug_shift_comp_baseline_new_lidar_precomp_fov_sparse_lidar_minmax_min
python3 main.py exp=depth_estimation/${exp}
```
