export MASTER_IP=$ARNOLD_WORKER_0_HOST
export MASTER_ADDR=$ARNOLD_WORKER_0_HOST
export MASTER_PORT=$ARNOLD_WORKER_0_PORT
export NODE_SIZE=$ARNOLD_WORKER_NUM
export NODE_RANK=$ARNOLD_ID
export HYDRA_FULL_ERROR=1
echo $MASTER_IP
python3 main.py exp=mde/mde_upsample_baseline_0404_aug_simu_linear_byte pl_trainer.devices=8 data.loader_opts.train.batch_size=6 +pl_trainer.num_nodes=4 exp_name=test4nodes
echo finished