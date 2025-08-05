import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/Users/linhaotong/Downloads/haoyu')
    parser.add_argument('--input_ply', type=str, default='/Users/linhaotong/Downloads/haoyu/evaluation-ours-20/105/dtu_3views_10.ply')
    args = parser.parse_args()
    return args

def main(args):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(args.input_ply)
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    height = 1200
    width = 1600
    vis.create_window(width=width, height=height) 
    # 将点云添加到Visualizer中
    vis.add_geometry(pcd)

    view_ctl = vis.get_view_control()
    extrinsic = np.array([
        [0.671673, -0.615308, 0.412615, -294.21],
        [0.24538, 0.710283, 0.659762, -402.82],
        [-0.699031, -0.341897, 0.628062, 247.354],
        [0.0, 0.0, 0.0, 1.0]
    ])
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=width,
        height=height,
        fx=2892.33,
        fy=2883.18,
        cx=width/2.,
        # cx=823.205,
        # cy=619.072,
        cy=height/2.
    )

    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic
    camera_params.extrinsic = extrinsic

    import ipdb; ipdb.set_trace()
    view_ctl.convert_from_pinhole_camera_parameters(camera_params)


    # 设置背景颜色为白色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])

    import ipdb; ipdb.set_trace()

    # 渲染图像并获取图像
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(True)
    # 将图像保存为带透明度的PNG文件
    o3d.io.write_image("output_image.png", o3d.geometry.Image((np.asarray(image) * 255).astype(np.uint8)))
    # 关闭Visualizer
    vis.destroy_window()

if __name__ == '__main__':
    args = parse_args()
    main(args)