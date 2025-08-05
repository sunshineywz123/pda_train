scenes=(
     "7b6477cb95" "c50d2d1d42" "cc5237fd77" "acd95847c5"
     "31a2c91c43" "e7af285f7d" "286b55a2bf" "7bc286c1b6"
)
# scenes=(
#     "c50d2d1d42" "cc5237fd77" "acd95847c5"
#     "31a2c91c43" "e7af285f7d" "286b55a2bf" "7bc286c1b6"
# )
# for scene in "${scenes[@]}"; do
#     export scene=$scene
#     python3 main.py exp=depth_estimation/marigold entry=val exp_name=marigold data=depth_estimation/scannetpp entry=val +model.output_tag=${scene} pl_trainer.devices=8 model.save_orig_pred=True data.dataset_opts.val.ensure_multiple_of=8 data.dataset_opts.val.scene=${scene} pl_trainer.limit_val_batches=1.
#     export exp=marigold
#     python3 scripts/utils/transfer_data.py
#     python3 scripts/utils/fuse_mesh.py --config_json /mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark/${scene}_${exp}/config.json
# done
# for scene in "${scenes[@]}"; do
#     export scene=$scene
#     export exp=may_depthanythingmetric_arkitscenes_hypersim_minmax
#     python3 main.py exp=depth_estimation/depth_anything entry=val exp_name=depth_anything_v2 data=depth_estimation/scannetpp entry=val +model.output_tag=${scene} pl_trainer.devices=8 model.save_orig_pred=True data.dataset_opts.val.ensure_multiple_of=14 data.dataset_opts.val.scene=${scene} pl_trainer.limit_val_batches=1.
#     python3 scripts/utils/transfer_data.py
#     python3 scripts/utils/fuse_mesh.py --config_json /mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark/${scene}_${exp}/config.json
# done
# for scene in "${scenes[@]}"; do
#     export scene=$scene
#     export exp=may_depthanythingmetric_arkitscenes_hypersim_minmax
#     export is_disparity=False
#     python3 main.py exp=depth_estimation/${exp} entry=val data=depth_estimation/scannetpp entry=val +model.output_tag=${scene} pl_trainer.devices=8 model.save_orig_pred=True data.dataset_opts.val.ensure_multiple_of=14 data.dataset_opts.val.scene=${scene} pl_trainer.limit_val_batches=1.
#     python3 scripts/utils/transfer_data.py
#     python3 scripts/utils/fuse_mesh.py --config_json /mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark/${scene}_${exp}/config.json
# done
for scene in "${scenes[@]}"; do
    export scene=$scene
    export exp=lowres_lidar_bilinear_upsample
    python3 scripts/utils/fuse_mesh.py --config_json /mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark/${scene}_${exp}/config.json
    # export exp=lowres_lidar_bilinear_upsample
    # python3 scripts/utils/fuse_mesh.py --config_json /mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark/${scene}_${exp}/config.json
done