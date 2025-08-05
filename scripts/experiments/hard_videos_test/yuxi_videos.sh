videos=("bmx-trees" "dog" "cr7" "horsejump" "parkour")
for video in "${videos[@]}"; do
    python main.py exp=marigold/marigold_official entry=val +model.pipeline.args.repeat_num=10 data=dpt/example_folder save_tag=${video} data.dataset_opts.val.data_root=data/pl_htcode/hard_videos/${video}
    python main.py exp=marigold/marigold_disp entry=val exp_name=marigold_disp_8gpus +model.pipeline.args.repeat_num=10 data=dpt/example_folder save_tag=${video} data.dataset_opts.val.data_root=data/pl_htcode/hard_videos/${video}
    python scripts/marigold/marigold2video.py --input data/pl_htcode/outputs/marigold/marigold_official/results/${video} --output data/pl_htcode/outputs/marigold/marigold_official/results/${video}/output.mp4
    python scripts/marigold/marigold2video.py --input data/pl_htcode/outputs/marigold/marigold_disp_8gpus/results/${video} --output data/pl_htcode/outputs/marigold/marigold_disp_8gpus/results/${video}/output.mp4
done
