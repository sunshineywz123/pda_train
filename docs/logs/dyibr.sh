function process_3d_scene_kid {
    local images_input_dir=$1
    local colmap_input_dir=$2
    local output_dir=$3

    local depth_methods=("zoedepth" "midas" "depth_anything" "marigold")
    local use_disp_flag=("" "--use_disp" "--use_disp" "")
    # 定义基本路径
    # local base_path="/mnt/remote/D001/Datasets/DL3DV/${scene_id}/nerfstudio"
    # local colmap_path="${base_path}/colmap/sparse/0"
    local output_base_path="data/pl_htcode/3d_fusion/dyibar_${scene_id}"
    for i in "${!depth_methods[@]}"; do
        local method=${depth_methods[$i]}
        local disp_flag=${use_disp_flag[$i]}
        local depth_output="${output_dir}/${method}"
        # 生成深度图
        proxy python scripts/${method}/estimate_folder.py --input ${images_input_dir} --output ${depth_output} ${disp_flag}
        # 深度融合
        python scripts/fusion_reldepth/run.py --colmap_input ${colmap_input_dir} --depth_input ${depth_output} --output ${depth_output}/fusion --rgb_input ${images_input_dir} ${disp_flag}
    done
}

process_3d_scene_kid data/pl_htcode/dynibar_release/kid-running/dense/images_512x288 data/pl_htcode/dynibar_release/kid-running/dense/sparse/0 data/pl_htcode/3d_fusion/dyibar_kid
