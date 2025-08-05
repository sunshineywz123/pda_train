export WORKSPACE=/mnt/bn/haotongdata/test/BD_Test/RenderingOutput/CameraComponent
export DB_PATH=$WORKSPACE/database.db

echo "WORKSPACE: $WORKSPACE"
echo "DB_PATH: $DB_PATH"

# colmap feature_extractor \
#     --database_path $DB_PATH \
#     --image_path $WORKSPACE/ColorImage \
#     --ImageReader.camera_model PINHOLE \
#     --ImageReader.single_camera 1

# colmap exhaustive_matcher \
#     --database_path $DB_PATH

mkdir -p $WORKSPACE/sparse

# colmap mapper \
#     --database_path $DB_PATH \
#     --image_path $WORKSPACE/ColorImage \
#     --output_path $WORKSPACE/sparse 

mkdir -p $WORKSPACE/sparse/0_txt
colmap model_converter \
    --input_path $WORKSPACE/sparse/0 \
    --output_path $WORKSPACE/sparse/0_txt \
    --output_type TXT

mkdir -p $WORKSPACE/sparse/0_aligned
colmap model_orientation_aligner \
    --input_path $WORKSPACE/sparse/0 \
    --output_path $WORKSPACE/sparse/0_aligned \
    --image_path $WORKSPACE/ColorImage