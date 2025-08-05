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
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    Log.info("num vertices raw {}".format(len(mesh.vertices)))
    Log.info("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def main(input_path, depth_path_dir, exp, tag, save_tag, scene, use_post=False, use_colmap_undistort=True):
    if use_colmap_undistort:
        colmap_path = 'colmap/dense/sparse'
    else:
        colmap_path = 'colmap/sparse_render_rgb'
    input_h, input_w = 756, 1008
    max_depth = 5.
    colmap_path = join(input_path, colmap_path)
    image_path = join(input_path, 'images')
    cams, images, points3d = read_model(colmap_path)

    if save_tag is None: mesh_save_path = join(input_path, f'plys/{exp}_{tag}.ply')
    else: mesh_save_path = join(input_path, f'plys/{exp}_{tag}_{save_tag}.ply')

    if exp == 'lidar':
        if save_tag is None: mesh_save_path = join(input_path, f'plys/lidar_tsdf.ply')
        else: mesh_save_path = join(input_path, f'plys/lidar_tsdf_{save_tag}.ply')

    if exp == 'gt':
        if save_tag is None: mesh_save_path = join(input_path, f'plys/gt_tsdf.ply')
        else: mesh_save_path = join(input_path, f'plys/gt_tsdf_{save_tag}.ply')

    os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
    name2id = {image.name:k for k, image in images.items()}
    names = [image.name for k, image in images.items()]
    names = sorted(names)
    names = [name for name in names if 'iphone' in name]

    voxel_size = 0.04
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02,
        sdf_trunc=0.15,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for idx, name in tqdm(enumerate(names)):
        if 'render_rgb' in name: continue
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
        ixt[:2, 2] = ixt[:2, 2] - 0.5
        if camera.model == 'OPENCV':
            dist = np.zeros(5)
            dist[:4] = camera.params[4:]
            orig_txt = ixt.copy()
            ixt, roi = cv2.getOptimalNewCameraMatrix(ixt, dist, (cam_width, cam_height), 1, (cam_width, cam_height))
            orig_ixt_new = ixt.copy()
        ixt[:2, 2] = ixt[:2, 2] + 0.5
        ixt[:1] *= (input_w / cam_width)
        ixt[1:2] *= (input_h / cam_height)
        extrinsic = ext
        extrinsic = o3c.Tensor(extrinsic, o3d.core.Dtype.Float64)

        if exp == 'lidar':
            depth_path_dir = join(input_path.replace('merge_dslr_iphone', 'iphone'), 'depth')
            depth_path_item = join(depth_path_dir, f'{frame_name}.png')
        elif exp == 'gt':
            depth_path_dir = join(input_path, 'render_depth')
            depth_path_item = join(depth_path_dir, f'{frame_name}.png')
        else:
            # depth_path_item = join(depth_path_dir, f'{frame_name}.npz')
            depth_path_item = join(depth_path_dir, 'orig_pred', f'{scene}__merge_dslr_iphone__images__iphone__{frame_name}.npz')
        if not os.path.exists(depth_path_item):
            continue
        if depth_path_item.endswith('.npz'):
            depth = np.load(depth_path_item)['data']
        elif depth_path_item.endswith('.png'):
            depth = imageio.imread(depth_path_item) / 1000.
        else:
            raise ValueError(f'Unknown depth format: {depth_path_item}')
        if depth.shape[0] != input_h or depth.shape[1] != input_w:
            # Log.warn(f'Invalid depth shape: {depth.shape}')
            depth = cv2.resize(depth, (input_w, input_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        intrinsic = o3c.Tensor(ixt[:3, :3], o3d.core.Dtype.Float64)

        img = cv2.imread(join(image_path, name))
        if camera.model == 'OPENCV' and use_colmap_undistort <= 0:
            img = cv2.undistort(img, orig_txt, dist, None, orig_ixt_new)
            img = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)
    Log.info(f'Save to {mesh_save_path}')
    pcd = o3d.geometry.PointCloud(mesh.vertices)
    if use_post > 0:
        mesh_post = post_process_mesh(mesh, cluster_to_keep=50)
        mesh_save_path = mesh_save_path.replace('.ply', '_post.ply')
        o3d.io.write_triangle_mesh(mesh_save_path, mesh_post)
        Log.info(f'Save to {mesh_save_path}')
        pcd_post = o3d.geometry.PointCloud(mesh_post.vertices)
        pcd = pcd_post
    gt_mesh = o3d.io.read_triangle_mesh(join(input_path.replace('merge_dslr_iphone', 'scans'), 'mesh_aligned_0.05.ply'))
    gt_pcd = o3d.geometry.PointCloud(gt_mesh.vertices)
    metrics = evaluate(pcd, gt_pcd)
    return metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone')
    parser.add_argument('--colmap_path', type=str, default='colmap/sparse_render_rgb')
    parser.add_argument('--exp', type=str, default='july_hypersim_baseline0717_new')
    parser.add_argument('--exp_tag', type=str, default='0703')
    parser.add_argument('--save_tag', type=str, default=None)
    parser.add_argument('--use_post', type=int, default=0)
    parser.add_argument('--use_colmap_undistort', type=int, default=0)
    parser.add_argument('--test_one_scene', action='store_true')
    parser.add_argument('--input_h', type=int, default=756)
    parser.add_argument('--input_w', type=int, default=1008)
    parser.add_argument('--max_depth', type=float, default=5)
    parser.add_argument('--min_depth', type=float, default=0.25)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    scenes = [
    "09c1414f1b", # "31a2c91c43",
    "1ada7a0617",
    "40aec5fffa",
    "3e8bba0176",# "e7af285f7d",
    "acd95847c5",
    "578511c8a9",
    "5f99900f09",
    "c4c04e6d6c",
    "f3d64c30f8",
    "7bc286c1b6",
    "c5439f4607",
    "286b55a2bf",
    "fb5a96b1a2"]
    if args.test_one_scene:
        scene = '5f99900f09'
        input_path = '/mnt/bn/haotongdata/Datasets/scannetpp/data/' + scene + '/merge_dslr_iphone'
        exp = args.exp
        exp_tag = args.exp_tag
        depth_path_dir = f'/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/{exp}/results/{scene}_{exp_tag}/orig_pred'
        tag = f'{scene}_{exp_tag}'
        args.input_path = input_path
        main(args, depth_path_dir, exp, tag, args.save_tag)
        sys.exit(0)
    
    def process_one_scene(scene):
        input_path = '/mnt/bn/haotongdata/Datasets/scannetpp/data/' + scene + '/merge_dslr_iphone'
        exp = args.exp
        exp_tag = args.exp_tag
        if args.use_colmap_undistort:
            depth_path_dir = f'/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/{exp}/results/scannetpp_colmap_undistort'
        else:
            depth_path_dir = f'/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/{exp}/results/scannetpp'
        tag = f'{scene}_{exp_tag}'
        metrics = main(input_path, depth_path_dir, exp, tag, args.save_tag, scene, args.use_post, args.use_colmap_undistort)
        return metrics


    # scenes = scenes[:3]
    # process_one_scene(scenes[1])
    # import ipdb; ipdb.set_trace()
    # process_one_scene(scenes[0])
    metrics = parallel_execution(
        scenes,
        action=process_one_scene,
        print_progress=True,
        desc='Processing scenes',
    )
    # print(metrics)

    exp = args.exp
    results = {scene: metric for scene, metric in zip(scenes, metrics)}
    mean = {k:np.mean([results[scene][k] for scene in results]) for k in metrics[0].keys()} 
    results['mean']  = mean
    if args.save_tag is not None: output_path = f'data/pl_htcode/txts/scannetpp/{exp}_{args.save_tag}.json'
    else: output_path = f'data/pl_htcode/txts/scannetpp/{exp}.json'
    lines = []
    for scene_name, metrics in results.items():
        line = json.dumps({scene_name: metrics})
        lines.append(line)
    with open(output_path, 'w') as f:
        f.write('[\n')
        for line in lines:
            f.write('    ' + line + ',\n')
        f.write(']\n')
    Log.info(f'Save to {output_path}')