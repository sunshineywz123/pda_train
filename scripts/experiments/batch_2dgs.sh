# /bin/zsh
mlx_submit_v2() {
  if [[ -e scripts/mlx_submit/run_v2.py ]]; then
    python3 scripts/mlx_submit/run_v2.py "$@"
  else
    echo "错误：scripts/mlx_submit/run_v2.py 不存在，请确认路径是否正确。"
  fi
}

mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_10.json
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_15.json
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_20.json
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_25.json
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_30.json
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_40.json
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_50.json
export DEPTH_DIR=/mnt/bn/haotongdata/Datasets/mycapture_arkit/20240629_centerflower/depth  
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_10.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_15.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_20.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_25.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_30.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_40.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_50.json $DEPTH_DIR
export DEPTH_DIR=/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/july_hypersim_scannetpp_zip_mesh/results/20240629_centerflower_0708_undistort_beforenet/orig_pred
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_10.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_15.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_20.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_25.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_30.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_40.json $DEPTH_DIR
mlx_submit_v2 --job_name test_2dgs --num_mem 40 --num_gpus 1 bash scripts/run_2dgs.sh 20240629_centerflower fps_split_50.json $DEPTH_DIR
