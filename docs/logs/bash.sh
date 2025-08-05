# 1de58be515696102c364b767f296600ffff853d4145a60dd30ece9d935317654
# 2beaca318994c25409dcbb6d0bdd96c3620f2f18aec44ea3f20edd302f18ca78
function process_3d_scene {
    local scene_id=$1
    local depth_methods=("zoedepth" "midas" "depth_anything" "marigold")
    local use_disp_flag=("" "--use_disp" "--use_disp" "")
    # 定义基本路径
    local base_path="/mnt/remote/D001/Datasets/DL3DV/${scene_id}/nerfstudio"
    local images_path="${base_path}/images_8"
    local colmap_path="${base_path}/colmap/sparse/0"
    local output_base_path="data/pl_htcode/3d_fusion/dl3dv_${scene_id}"
    for i in "${!depth_methods[@]}"; do
        local method=${depth_methods[$i]}
        local disp_flag=${use_disp_flag[$i]}
        local depth_output="${output_base_path}/${method}"
        # 生成深度图
        proxy python scripts/${method}/estimate_folder.py --input ${images_path} --output ${depth_output} ${disp_flag}
        # 深度融合
        python scripts/fusion_reldepth/run.py --colmap_input ${colmap_path} --depth_input ${depth_output} --output ${depth_output}/fusion --rgb_input ${images_path} ${disp_flag}
    done
}
scene_ids=("2beaca318994c25409dcbb6d0bdd96c3620f2f18aec44ea3f20edd302f18ca78" "3bb894d1933f3081134ad2d40e54de5f0636bd8b502b0a8561873bb63b0dce85" "7da3db99059a436adf537bc23309a6a4db9e011696d086f3a64037f7497d9df7" "513e4ea2e8477b06f2b32417d5c243aec71d491cb0596a60e9fe304c635f20a1" "7705a2edd022de9da3f8cead677287645986e982d9247a21e1992859a59f8335" "/mnt/remote/D001/Datasets/DL3DV/b4f53094fd31dcc2de1c1d1fdfeffac5259e9bdf39d77b552ff948e4bdf1fd8e" "ef59aac437132bfc1dd45a7e1f8e4800978e7bb28bf98c4428d26fb3e1da3e90" "fb2c0499c225d6124938cadcf0dc48cbb490551b8a69c98386491c1366163632" "ff592398657b3dfe94153332861985194a3e3c9d199c4a3a27a0ce4038e81ade" "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7")
for scene_id in "${scene_ids[@]}"; do
    echo "Starting processing for scene: ${scene_id}"
    process_3d_scene ${scene_id}
    echo "Finished processing for scene: ${scene_id}"
done
