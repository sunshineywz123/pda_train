cd /mnt/bn/haotongdata/home/linhaotong/projects/zipnerf-pytorch
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <value1>"
    exit 1
fi
export scene="$1"
export DATA_DIR=/mnt/bn/haotongdata/Datasets/scannetpp/data/${scene}/merge_dslr_iphone
export EXP_NAME=scannetpp_all_0610/${scene}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export colmap_dir=colmap/sparse/0
echo $scene
accelerate launch --main_process_port=9999 train.py --gin_configs=configs/360_glo_scannetpp.gin --gin_bindings="Config.colmap_dir = '${colmap_dir}'" --gin_bindings="Config.data_dir = '${DATA_DIR}'"  --gin_bindings="Config.exp_name = '${EXP_NAME}'" --gin_bindings="Config.factor = -1" --gin_bindings="Config.llff_use_all_images_for_training = True" --gin_bindings="Config.max_steps = 50000"
accelerate launch --main_process_port=9999 eval.py --gin_configs=configs/360_glo_scannetpp.gin --gin_bindings="Config.colmap_dir = '${colmap_dir}'" --gin_bindings="Config.data_dir = '${DATA_DIR}'" --gin_bindings="Config.exp_name = '${EXP_NAME}'" --gin_bindings="Config.factor = -1" --gin_bindings="Config.llff_use_all_images_for_testing = True"
cd $ICCV_CODE_PATH