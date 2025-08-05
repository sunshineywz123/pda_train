
## debug train dataloader

```
python main.py exp=mde/mde entry=debug_train_dataloader data.loader_opts.train.num_workers=0
```

## Entrys

1. visualization for temporal stablity, human
2. Reconstruct statue



## 1. Visualization for temporal stablity, human

### setup exp and expname
```
export exp=aug_hypersim_arkit_random_all_dataset 
export exp_name=aug_hypersim_arkit_random_all_dataset_zip_dydtof
# yoga
export dataset=depth_estimation/mycapture_v3
export scene=seq4
export output_tag=yoga_$scene
```

```
# predict depth
python3 main.py exp=depth_estimation/${exp} exp_name=${exp_name} data=${dataset} entry=predict \
data.val_dataset.dataset_opts.frames=\[5300,6500,1\] \
data.val_dataset.dataset_opts.scene=${scene} \
+model.output_tag=${output_tag} \
+model.clear_output_dir=True \
model.save_orig_pred=True +model.near_depth=0.5 +model.far_depth=5. +data.val_dataset.dataset_opts.use_colmap=False 
```

```
# save pcd to video
export rgb_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/${output_tag}/vis_depth   
export depth_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/${output_tag}/orig_pred   
export frame_start=0
export frame_end=1200
export tar_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/${output_tag}/vis_pcd_${frame_start}_${frame_end}
python3 scripts/mycapture/gen_pcd_video.py --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --tar_dir ${tar_dir} --frame_start ${frame_start} --frame_end ${frame_end}
export tar_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/${output_tag}/vis_pcd_${frame_start}_${frame_end}/rgb.mp4
```
```
# save lowres pcd to video
export depth_dir=/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/seq4/depth
export frame_start=0
export frame_end=1200
export tar_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/${output_tag}/vis_lowres_pcd_${frame_start}_${frame_end}
python3 scripts/mycapture/gen_pcd_video.py --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --tar_dir ${tar_dir} --frame_start ${frame_start} --frame_end ${frame_end}
export tar_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/${output_tag}/vis_pcd_${frame_start}_${frame_end}/rgb.mp4

```


## 2. Reconstruct statue

```
export exp=aug_hypersim_arkit_random_all_dataset 
export exp_name=aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask
```

### 20240702_statue1

```
export data_root=/mnt/bn/haotongdata/Datasets/mycapture_arkit
export scene=20240629_statue1
export scene=20240702_statue1
export scene=20240702_statue2
export scene=20240702_statue3
export scene=20240702_statue4
export data_root=/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner
export scene=e4b760e009 # 
export scene=cff3fcbcb0 # yujie half body
export scene=ede404347d # haotong outdoor
export scene=85154946bc # 
export scene=48f8b59e4f # objects, middle
export scene=56ef6962e8 # objects, near
export scene=d1bb5a1af4  # objects, 

python3 main.py exp=depth_estimation/${exp} exp_name=${exp_name} entry=predict data=depth_estimation/mycapture_v3 data.val_dataset.dataset_opts.data_root=${data_root} data.val_dataset.dataset_opts.scene=${scene} +data.val_dataset.dataset_opts.undistort_colmap=True +model.output_tag=${scene} model.save_orig_pred=True

export colmap_model=${data_root}/${scene}/colmap/dense/sparse
export rgb_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/${scene}/vis_depth
export depth_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/${scene}/orig_pred
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/${scene}/fuse.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path} --max_depth 4
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/${scene}/fuse_0.01.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path} --max_depth 4 --voxel_size 0.01
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/${scene}/fuse_0.005.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path} --max_depth 4 --voxel_size 0.005


export depth_dir=${data_root}/${scene}/depth
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/${scene}/fuse_lidar.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path} --max_depth 4
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/${scene}/fuse_lidar_0.01.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path} --max_depth 4 --voxel_size 0.01
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/${scene}/fuse_lidar_0.005.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path} --max_depth 4 --voxel_size 0.005
```





## 可视化点云

## 可视化重建

## 评估重建

## 评估上采样



## inference depth

### evaluate ours depth on scannetpp and pcd video visualization
```
export exp=aug_hypersim_arkit_random_all_dataset
export exp_name=${exp}_zip_grad0.5_new_mask
export scene=5f99900f09
python3 main.py exp=depth_estimation/${exp} exp_name=${exp_name} data=depth_estimation/mycapture_v3 entry=predict data.val_dataset.dataset_opts.frames=\[0,-1,1\] data.val_dataset.dataset_opts.scene=iphone +model.output_tag=scannetpp_${scene}_video_new model.save_orig_pred=True +model.near_depth=0.25 +model.far_depth=6. +data.val_dataset.dataset_opts.use_colmap=False data.val_dataset.dataset_opts.data_root=data/pl_htcode/datasets/scannetpp/data/${scene}

export frame_start=3000
export frame_end=4200
export rgb_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_new/vis_depth
export depth_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_new/orig_pred
export tar_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_new/pcd_video_${frame_start}_${frame_end}
python3 scripts/mycapture/gen_pcd_video.py --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --tar_dir ${tar_dir} --frame_start ${frame_start} --frame_end ${frame_end}



export frame_start=3000
export frame_end=4200
export rgb_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_new/vis_depth
export depth_dir=/mnt/bn/haotongdata/Datasets/scannetpp/data/${scene}/iphone/depth
export tar_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_new/pcd_video_lowres_${frame_start}_${frame_end}
python3 scripts/mycapture/gen_pcd_video.py --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --tar_dir ${tar_dir} --frame_start ${frame_start} --frame_end ${frame_end} --depth_format .png




export exp=aug_zipmesh_arkit
export exp_name=${exp}_release
export scene=5f99900f09
python3 main.py exp=depth_estimation/${exp} exp_name=${exp_name} data=depth_estimation/mycapture_v3 entry=predict data.val_dataset.dataset_opts.frames=\[0,-1,1\] data.val_dataset.dataset_opts.scene=iphone +model.output_tag=scannetpp_${scene}_video_new model.save_orig_pred=True +model.near_depth=0.25 +model.far_depth=6. +data.val_dataset.dataset_opts.use_colmap=False data.val_dataset.dataset_opts.data_root=data/pl_htcode/datasets/scannetpp/data/${scene} +model.fit_to_tar=True
```

### lowres lidar pcd video visualization

### depth anything visualization and pcd video visualization

### scannetpp ours reconstruction 

```
export exp=aug_hypersim_arkit_random_all_dataset
export exp_name=${exp}_zip_grad0.5_new_mask
export scene=5f99900f09
export split=split_0
python3 main.py exp=depth_estimation/${exp} exp_name=${exp_name} data=depth_estimation/mycapture_v3 entry=predict data.val_dataset.dataset_opts.frames=\[0,-1,1\] data.val_dataset.dataset_opts.scene=${split} +model.output_tag=scannetpp_${scene}_video_for_recon_${split} model.save_orig_pred=True +model.near_depth=0.25 +model.far_depth=6. +data.val_dataset.dataset_opts.use_colmap=True data.val_dataset.dataset_opts.data_root=/mnt/bn/haotongdata/Datasets/scannetpp/data/${scene}/iphone/split_output +data.val_dataset.dataset_opts.undistort_colmap=True

export colmap_model=/mnt/bn/haotongdata/Datasets/scannetpp/data/${scene}/iphone/split_output/${split}/colmap/dense/sparse 
export rgb_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/vis_depth
export depth_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/orig_pred
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/fuse.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path}
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/fuse_0.01.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path} --voxel_size 0.01
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/fuse_0.01_10.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path} --voxel_size 0.01 --trunc_num 10



export colmap_model=/mnt/bn/haotongdata/Datasets/scannetpp/data/${scene}/iphone/split_output/${split}/colmap/dense/sparse 
export rgb_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/vis_depth
export depth_dir=/mnt/bn/haotongdata/Datasets/scannetpp/data/${scene}/iphone/split_output/${split}/depth
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/fuse_lidar.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path}
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/fuse_lidar_0.01.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path} --voxel_size 0.01



```

### scannetpp depth anything reconstruction 
```
export exp=aug_zipmesh_arkit
export exp_name=${exp}_release
export scene=09c1414f1b
export split=split_2
python3 main.py exp=depth_estimation/${exp} exp_name=${exp_name} data=depth_estimation/mycapture_v3 entry=predict data.val_dataset.dataset_opts.frames=\[0,-1,1\] data.val_dataset.dataset_opts.scene=${split} +model.output_tag=scannetpp_${scene}_video_for_recon_${split} model.save_orig_pred=True +model.near_depth=0.25 +model.far_depth=6. +data.val_dataset.dataset_opts.use_colmap=True data.val_dataset.dataset_opts.data_root=/mnt/bn/haotongdata/Datasets/scannetpp/data/${scene}/iphone/split_output +data.val_dataset.dataset_opts.undistort_colmap=True +model.fit_to_tar=True



export colmap_model=/mnt/bn/haotongdata/Datasets/scannetpp/data/${scene}/iphone/split_output/${split}/colmap/dense/sparse 
export rgb_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/vis_depth
export depth_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/orig_pred
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/fuse.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path}
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_${split}/fuse_0.01.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path} --voxel_size 0.01

python3 main.py exp=depth_estimation/${exp} exp_name=${exp_name} data=depth_estimation/mycapture_v3 entry=predict data.val_dataset.dataset_opts.frames=\[0,-1,1\] data.val_dataset.dataset_opts.scene=${split} +model.output_tag=scannetpp_${scene}_video_for_recon_${split} model.save_orig_pred=True +model.near_depth=0.25 +model.far_depth=6. +data.val_dataset.dataset_opts.use_colmap=True data.val_dataset.dataset_opts.data_root=/mnt/bn/haotongdata/Datasets/scannetpp/data/${scene}/iphone/split_output +data.val_dataset.dataset_opts.undistort_colmap=True +model.fit_to_tar=True

export colmap_model=/mnt/bn/haotongdata/Datasets/scannetpp/data/${scene}/iphone/split_output/split_0/colmap/dense/sparse 
export rgb_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_fit/vis_depth
export depth_dir=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_fit/orig_pred
export output_path=${workspace}/outputs/depth_estimation/${exp_name}/results/scannetpp_${scene}_video_for_recon_fit/fuse.ply
python3 scripts/scannetpp/fuse_pred.py --colmap_model ${colmap_model} --rgb_dir ${rgb_dir} --depth_dir ${depth_dir} --output_path ${output_path}
```


## lidar

### all in one 

```
scenes=(036 140 148 157 181 226 232 237 241 245 271 297 302 312 314 362 482 495 524 527 780)
for scene in "${scenes[@]}"; do
    echo "Processing scene $scene"
    export scene="$scene"
    # Navigate to the project directory and generate lidar depth
    cd /mnt/bn/haotongdata/home/linhaotong/projects/street_gaussians
    python3 script/waymo/generate_lidar_depth.py --datadir /mnt/bn/haotongdata3/Datasets/waymo/${scene}
    # Navigate to the ICCV code path and generate splits
    cd $ICCV_CODE_PATH
    python3 scripts/waymo/generate_splits.py --seq ${scene}

    # Set experiment variable and execute main.py with specified parameters
    export exp="aug_shift_comp_baseline_new_lidar_precomp_fov_sparse_lidar_minmax_min"
    python3 main.py exp=depth_estimation/${exp} data=depth_estimation/waymo_comp +model.output_tag=waymo_${scene}_756 entry=val exp_name=${exp} +model.near_depth=1. +model.far_depth=50. data.val_dataset.dataset_opts.split_path=data/pl_htcode/processed_datasets/waymo/${scene}.json model.save_orig_pred=True

    # Generate depth estimation video
    ffmpeg -framerate 20 -i  ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/vis_depth/${scene}__images__%06d_0.jpg -c:v libx264 -r 10 -pix_fmt yuv420p ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/output.mp4

    # Navigate to BP-Net project directory and perform tests
    cd /mnt/bn/haotongdata/home/linhaotong/projects/BP-Net
    python3 test.py gpus=\[0\] name=BP_KITTI ++chpt=BP_KITTI net=PMP num_workers=4 data=KITTI data.testset.mode=waymo_${scene} data.testset.height=352 test_batch_size=1 metric=RMSE ++save=true

    # Generate BP-Net video
    ffmpeg -framerate 20 -i /mnt/bn/haotongdata/home/linhaotong/projects/BP-Net/results/BP_KITTI/waymo_${scene}/%010d.jpg -c:v libx264 -r 10 -pix_fmt yuv420p ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/bpnet.mp4

    # Concatenate videos side-by-side
    ffmpeg -i ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/output.mp4 -i ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/bpnet.mp4 -filter_complex "[1:v]scale=756:504[resized];[0:v][resized]hstack=inputs=2" -c:v libx264 ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/output_concat.mp4
done
echo "All scenes processed."
```

```
export scene=200
cd /mnt/bn/haotongdata/home/linhaotong/projects/street_gaussians
python3 script/waymo/generate_lidar_depth.py --datadir /mnt/bn/haotongdata3/Datasets/waymo/${scene}
cd $ICCV_CODE_PATH
python3 scripts/waymo/generate_splits.py --seq ${scene}
export exp=aug_shift_comp_baseline_new_lidar_precomp_fov_sparse_lidar_minmax_min
python3 main.py exp=depth_estimation/${exp} data=depth_estimation/waymo_comp +model.output_tag=waymo_${scene}_756 entry=val exp_name=${exp} +model.near_depth=1. +model.far_depth=50. data.val_dataset.dataset_opts.split_path=data/pl_htcode/processed_datasets/waymo/${scene}.json
ffmpeg -framerate 20 -i  ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/vis_depth/${scene}__images__%06d_0.jpg  -c:v libx264 -r 10 -pix_fmt yuv420p ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/output.mp4
cd /mnt/bn/haotongdata/home/linhaotong/projects/BP-Net
python3 test.py gpus=\[0\] name=BP_KITTI ++chpt=BP_KITTI \
net=PMP num_workers=4 \
data=KITTI data.testset.mode=waymo_${scene} data.testset.height=352 \
test_batch_size=1 metric=RMSE ++save=true
ffmpeg -framerate 20 -i /mnt/bn/haotongdata/home/linhaotong/projects/BP-Net/results/BP_KITTI/waymo_${scene}/%010d.jpg  -c:v libx264 -r 10 -pix_fmt yuv420p ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/bpnet.mp4 
ffmpeg -i ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/output.mp4 -i ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/bpnet.mp4 \
-filter_complex "[1:v]scale=756:504[resized];[0:v][resized]hstack=inputs=2" \
-c:v libx264 ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/output_concat.mp4
```


### prepare waymo
```
export scene=019
cd /mnt/bn/haotongdata/home/linhaotong/projects/street_gaussians
python3 script/waymo/generate_lidar_depth.py --datadir /mnt/bn/haotongdata3/Datasets/waymo/${scene}
cd $ICCV_CODE_PATH
python3 scripts/waymo/generate_splits.py --seq ${scene}

```

###  inference ours

```
cd $ICCV_CODE_PATH
export scene=019
export exp=aug_shift_comp_baseline_new_lidar_precomp_fov_sparse_lidar_minmax_min
python3 main.py exp=depth_estimation/${exp} data=depth_estimation/waymo_comp +model.output_tag=waymo_${scene}_756 entry=val exp_name=${exp} +model.near_depth=1. +model.far_depth=50. data.val_dataset.dataset_opts.split_path=data/pl_htcode/processed_datasets/waymo/${scene}.json model.save_orig_pred=True
ffmpeg -framerate 20 -i  ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/vis_depth/${scene}__images__%06d_0.jpg  -c:v libx264 -r 10 -pix_fmt yuv420p ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/output.mp4
```

###  inference BP-Net
```
export scene=019
cd /mnt/bn/haotongdata/home/linhaotong/projects/BP-Net
python3 test.py gpus=\[0\] name=BP_KITTI ++chpt=BP_KITTI \
net=PMP num_workers=4 \
data=KITTI data.testset.mode=waymo_${scene} data.testset.height=352 \
test_batch_size=1 metric=RMSE ++save=true
ffmpeg -framerate 20 -i /mnt/bn/haotongdata/home/linhaotong/projects/BP-Net/results/BP_KITTI/waymo_${scene}/%010d.jpg  -c:v libx264 -r 10 -pix_fmt yuv420p ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/bpnet.mp4 
```

### concat
```
ffmpeg -i ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/output.mp4 -i ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/bpnet.mp4 \
-filter_complex "[1:v]scale=756:504[resized];[0:v][resized]hstack=inputs=2" \
-c:v libx264 ${workspace}/outputs/depth_estimation/${exp}/results/waymo_${scene}_756/output_concat.mp4
```
