import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import open3d as o3d
import open3d.core as o3c
import cv2
import copy
sys.path.append('.')
from lib.utils.pylogger import Log
from trdparties.colmap.read_write_model import read_model
import imageio


def post_process_mesh(mesh, cluster_to_keep=50):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    Log.info("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    Log.info("num vertices raw {}".format(len(mesh.vertices)))
    Log.info("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def main(args):
    exp = args.exp
    frame = int(args.frame)
    colmap_path = join(args.input, 'colmap/sparse/0')
    cameras, images, _ = read_model(colmap_path)

    voxel_size = 0.01
    tsdf_trunc = voxel_size * 5
    depth_trunc = 4.
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=tsdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for im_id in images:
        image = images[im_id]
        img_name = image.name
        img_name_num = int(img_name.split('.')[0])
        if img_name_num < 30:
            continue

        ext = np.eye(4)
        ext[:3, :3] = image.qvec2rotmat()
        ext[:3, 3] = image.tvec
        c2w = np.linalg.inv(ext)
        ixt = np.eye(3)
        cam_id = image.camera_id
        camera = cameras[cam_id]
        cam_height, cam_width = camera.height, camera.width
        ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = camera.params[:4]
        ixt = ixt.copy()
        ixt[:2, 2] = ixt[:2, 2] - 0.5
        dist = np.zeros(5)
        dist[:4] = camera.params[4:]
        orig_ixt = ixt.copy()
        ixt, roi = cv2.getOptimalNewCameraMatrix(ixt, dist, (cam_width, cam_height), 1, (cam_width, cam_height))
        orig_ixt_new = ixt.copy()
        ixt[:2, 2] = ixt[:2, 2] + 0.5
        img_path = join(args.input, 'images', '{:02d}'.format(img_name_num), '{:06d}.jpg'.format(frame))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.undistort(img, orig_ixt, dist, None, orig_ixt_new)
        img = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

        if exp == 'lidar':
            depth = imageio.imread(join(args.input, 'depth', '{:02d}'.format(img_name_num), '{:06d}.png'.format(frame))) / 1000
        else:
            depth = np.load(f'/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/{exp}/results/seq2_mocap/orig_pred/{img_name_num:02d}__{frame:06d}.npz')['data']
        depth_height, depth_width = depth.shape
        img_height, img_width = img.shape[:2]
        img = cv2.resize(img, (depth_width, depth_height), interpolation=cv2.INTER_LINEAR)
        ixt[:1] = ixt[:1] * depth_width / img_width
        ixt[1:2] = ixt[1:2] * depth_height / img_height

        extrinsic = ext
        extrinsic = o3c.Tensor(extrinsic, o3d.core.Dtype.Float64)
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth_width, depth_height, ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]
        )
        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(img.astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=depth_trunc, convert_rgb_to_intensity=False, depth_scale=1.,
        )
        volume.integrate(
            rgbd,
            camera_intrinsics,  # type: ignore
            ext,
        )
    mesh = volume.extract_triangle_mesh()
    pcd = volume.extract_point_cloud()
    mesh_post = post_process_mesh(mesh, cluster_to_keep=50)
    o3d.io.write_triangle_mesh( args.output.replace('EXP', exp).replace('FRAME', str(frame)), mesh_post)
    o3d.io.write_point_cloud( args.output.replace('EXP', exp).replace('FRAME', str(frame) + '_pcd.ply'), pcd)
    Log.info("Done")
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/seq2_raw/seq2_mocap')
    parser.add_argument('--exp', type=str, default='july_hypersim_baseline0717_minmax_human')
    parser.add_argument('--frame', type=int, default=4200)
    parser.add_argument('--output', type=str, default='EXP_FRAME.ply')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)