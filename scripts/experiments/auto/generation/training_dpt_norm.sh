export MASTER_IP=$ARNOLD_WORKER_0_HOST
export MASTER_ADDR=$ARNOLD_WORKER_0_HOST
export MASTER_PORT=$ARNOLD_WORKER_0_PORT
export NODE_SIZE=$ARNOLD_WORKER_NUM
export NODE_RANK=$ARNOLD_ID
export HYDRA_FULL_ERROR=1
python3 main.py exp=lidarerr/diffusion_only_dpt_normalize pl_trainer.limit_val_batches=0. pl_trainer.devices=8 +pl_trainer.num_nodes=4