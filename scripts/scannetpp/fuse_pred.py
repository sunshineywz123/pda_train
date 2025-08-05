import json
import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
import open3d as o3d
import open3d.core as o3c

from lib.utils.parallel_utils import parallel_execution
from lib.utils.pylogger import Log
sys.path.append('.')
import open3d as o3d
from scipy.optimize import root
import cv2
cv2.setNumThreads(0)

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# from copy import copy
import copy

from trdparties.colmap.read_write_model import read_cameras_binary, read_images_binary, read_model

from sklearn.neighbors import KDTree
def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances
def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    if isinstance(mesh_pred, str):
        mesh_pred = o3d.io.read_triangle_mesh(mesh_pred)
    if isinstance(mesh_trgt, str):
        mesh_trgt = o3d.io.read_triangle_mesh(mesh_trgt)
    # pcd_pred = o3d.geometry.PointCloud(mesh_pred.vertices)
    # pcd_trgt = o3d.geometry.PointCloud(mesh_trgt.vertices)
    pcd_pred = mesh_pred
    pcd_trgt = mesh_trgt

    aabb_a = pcd_trgt.get_axis_aligned_bounding_box()
    points_b = np.asarray(pcd_pred.points)
    inside_aabb = (points_b[:, 0] >= aabb_a.min_bound[0] - 0.1) & \
              (points_b[:, 0] <= aabb_a.max_bound[0] + 0.1) & \
              (points_b[:, 1] >= aabb_a.min_bound[1] - 0.1) & \
              (points_b[:, 1] <= aabb_a.max_bound[1] + 0.1) & \
              (points_b[:, 2] >= aabb_a.min_bound[2] - 0.1) & \
              (points_b[:, 2] <= aabb_a.max_bound[2] + 0.1)
    pcd_pred = pcd_pred.select_by_index(inside_aabb.nonzero()[0])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recall': recal,
        'F-score': fscore,
    }
    return metrics

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
    try:
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    except:
        n_cluster = np.sort(cluster_n_triangles.copy())[0]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    Log.info("num vertices raw {}".format(len(mesh.vertices)))
    Log.info("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def main(args):
    colmap_model = args.colmap_model
    rgb_dir = args.rgb_dir
    depth_dir = args.depth_dir
    mesh_save_path = args.output_path
    voxel_size = args.voxel_size
    min_depth, max_depth = args.min_depth, args.max_depth
    cams, images, points3d = read_model(colmap_model)
    os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
    name2id = {image.name:k for k, image in images.items()}
    names = [image.name for k, image in images.items()]
    names = sorted(names)
    # names = [name for name in names if 'iphone' in name]

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size*args.trunc_num,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    input_h, input_w = 756, 1008
    for idx, name in tqdm(enumerate(names)):
        frame_name = os.path.basename(name)[:-4]
        image = images[name2id[name]]
        ext = np.eye(4)
        ext[:3, :3] = image.qvec2rotmat()
        ext[:3, 3] = image.tvec
        c2w = np.linalg.inv(ext)
        ixt = np.eye(3)
        cam_id = image.camera_id
        camera = cams[cam_id]
        cam_height, cam_width = camera.height, camera.width
        ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = camera.params[:4]
        ixt = ixt.copy()
        ixt[:1] *= (input_w / cam_width)
        ixt[1:2] *= (input_h / cam_height)
        extrinsic = ext
        extrinsic = o3c.Tensor(extrinsic, o3d.core.Dtype.Float64)
        try: depth = np.load(join(depth_dir, name.replace('.jpg', '.npz')))['data']
        except: depth = np.asarray(imageio.imread(join(depth_dir, name.replace('.jpg', '.png')))) / 1000.
        if depth.shape[0] != input_h or depth.shape[1] != input_w:
            depth = cv2.resize(depth, (input_w, input_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        intrinsic = o3c.Tensor(ixt[:3, :3], o3d.core.Dtype.Float64)

        img = cv2.imread(join(rgb_dir, name))[:input_h, :input_w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != input_h or img.shape[1] != input_w:
            img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            input_w, input_h, ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]
        )
        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(img.astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=max_depth, convert_rgb_to_intensity=False, depth_scale=1.,
        )

        volume.integrate(
            rgbd,
            camera_intrinsics,  # type: ignore
            ext,
        )
    mesh = volume.extract_triangle_mesh()
    mesh = post_process_mesh(mesh, cluster_to_keep=1000)
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)
    Log.info(f'Save to {mesh_save_path}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_model', type=str)
    parser.add_argument('--rgb_dir', type=str)
    parser.add_argument('--depth_dir', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.02)
    parser.add_argument('--trunc_num', type=float, default=5)
    parser.add_argument('--max_depth', type=float, default=5)
    parser.add_argument('--min_depth', type=float, default=0.25)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)