
#!/bin/bash
scenes=(
    "f3685d06a9" "b0a08200c9" "825d228aec" "a980334473"
    "f2dc06b1d2" "5942004064" "25f3b7a318" "bcd2436daf"
    "f3d64c30f8" "0d2ee665be" "3db0a1c8f3" "ac48a9b736"
    "c5439f4607" "578511c8a9" "d755b3d9d8" "99fa5c25e1"
    "09c1414f1b" "5f99900f09" "9071e139d9" "6115eddb86"
    "27dd4da69e" "c49a8c6cff"
)
# 每次处理8个场景
batch_size=4
for ((i=0; i<${#scenes[@]}; i+=batch_size)); do
    # 为当前批次的每个场景启动一个后台任务
    for ((j=0; j<batch_size && (i+j)<${#scenes[@]}; j++)); do
        scene=${scenes[i+j]}
        # echo ${scene} ${j}
        python3 main.py exp=mde/mde_upsample_baseline_0404_aug_simu_linear_byte entry=val data=mde/hypersim_val_byte +model.args.save_vis_pred=True +model.args.save_orig_pred=True +model.args.save_align_pred=True save_tag=scannetpp_${scene}-1024 +model.args.fit_ransac=True data=mde/scannetpp_val_byte +model.compute_lowres_rel_metric=True +model.compute_highres_rel_metric=True data.dataset_opts.val.scene=${scene} pl_trainer.devices=\[${j}\] pl_trainer.limit_val_batches=1. exp_name=marigold&
    done

    # 等待当前批次的所有任务完成
    wait
done

echo "所有命令已完成执行。"