
#!/bin/bash
scenes=(
    "7b6477cb95" "c50d2d1d42" "cc5237fd77" "acd95847c5"
    "fb5a96b1a2" "a24f64f7fb" "1ada7a0617" "5eb31827b7"
    "3e8bba0176" "3f15a9266d" "21d970d8de" "5748ce6f01"
    "c4c04e6d6c" "7831862f02" "bde1e479ad" "38d58a7a31"
    "5ee7c22ba0" "f9f95681fd" "3864514494" "40aec5fffa"
    "13c3e046d7" "e398684d27" "a8bf42d646" "45b0dac5e3"
    "31a2c91c43" "e7af285f7d" "286b55a2bf" "7bc286c1b6"
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