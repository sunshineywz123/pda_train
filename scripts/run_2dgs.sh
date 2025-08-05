cp -r /mnt/bn/haotongdata/home/linhaotong/projects/2dgs_byte ~/
cd ~/2dgs_byte
bash env.sh
export SOURCE_DIR=/mnt/bn/haotongdata/Datasets/mycapture_arkit
export MODEL_DIR=/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output
if [ -n "$1" ]; then
    export SCENE=$1
fi
if [ -n "$2" ]; then
    export SPLIT_FILE=splits/$2
    # export SPLIT_FILE=splits/fps_split_5.json
fi
export SOURCE_PATH=${SOURCE_DIR}/${SCENE}
export SPLIT_NAME=$(basename ${SPLIT_FILE} .json | cut -d'_' -f3)
export MODEL_PATH=${MODEL_DIR}/mycapture_${SCENE}_split${SPLIT_NAME}
if [ -n "$3" ]; then
    export DEPTH_DIR=$3
    if [[ "$DEPTH_DIR" == *orig_pred ]]; then
        export MODEL_PATH=${MODEL_PATH}_depth
    else
        export MODEL_PATH=${MODEL_PATH}_lidar
    fi
fi
export COLMAP_PATH=colmap/sparse/0_metric
echo $MODEL_PATH
echo $SOURCE_PATH
python3 train.py --source_path $SOURCE_PATH \
                 --model_path $MODEL_PATH \
                 --colmap_path $COLMAP_PATH --sh_degree 1 --eval \
                 --train_test_split_file $SPLIT_FILE 
python3 render.py --source_path $SOURCE_PATH \
                  --model_path  $MODEL_PATH \
                  --colmap_path $COLMAP_PATH --sh_degree 1 --eval \
                  --train_test_split_file $SPLIT_FILE \
                  --skip_mesh --skip_train