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
import copy
from sklearn.neighbors import KDTree
sys.path.append('.')
from lib.utils.pylogger import Log
import open3d as o3d
from scipy.optimize import root
import cv2

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from trdparties.colmap.read_write_model import read_cameras_binary, read_images_binary, read_model

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

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
    


def evalute_ply_path(source_ply_path, target_ply_path, transform='icp'):
    source = o3d.io.read_point_cloud(source_ply_path)
    target = o3d.io.read_point_cloud(target_ply_path)
    if transform == 'icp':
        global_voxel_size = 0.125
        source_down, source_fpfh = preprocess_point_cloud(source, global_voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, global_voxel_size)
        result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                global_voxel_size)
        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     global_voxel_size, result_ransac)
        transformation = result_icp.transformation
    elif transform == 'precomputed':
        transformation = np.loadtxt(source_ply_path.replace(os.path.basename(source_ply_path), 'transformation.txt'))
    elif transform is None:
        transformation = np.eye(4)
    source = source.transform(transformation)
    return evaluate(source, target)


def main(args, exp, tag, scene, save_tag, transform):
    if save_tag is None: source_ply_path  = join(args.input_path, f'plys/{exp}_{tag}.ply')
    else: source_ply_path  = join(args.input_path, f'plys/{exp}_{tag}_{save_tag}.ply')
    if exp == 'lidar': 
        if save_tag is None: source_ply_path  = join(args.input_path, f'plys/lidar_tsdf.ply')
        else: source_ply_path  = join(args.input_path, f'plys/lidar_tsdf_{save_tag}.ply')
    elif exp == 'gt': 
        if save_tag is None: source_ply_path  = join(args.input_path, f'plys/gt_tsdf.ply')
        else: source_ply_path  = join(args.input_path, f'plys/gt_tsdf_{save_tag}.ply')
    
    target_ply_path = join(args.input_path.replace('merge_dslr_iphone', 'scans'), f'mesh_aligned_0.05.ply')
    metrics = evalute_ply_path(source_ply_path, target_ply_path, transform)
    return metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone')
    parser.add_argument('--exp', type=str, default='june_depthanythingmetric_scannetpp_0614_hypersim_mask_far')
    parser.add_argument('--exp_tag', type=str, default='0703')
    parser.add_argument('--test_one_scene', action='store_true')
    parser.add_argument('--save_tag', type=str, default=None)
    parser.add_argument('--transform', type=str, default=None)
    parser.add_argument('--input_h', type=int, default=756)
    parser.add_argument('--input_w', type=int, default=1008)
    parser.add_argument('--max_depth', type=float, default=5)
    parser.add_argument('--min_depth', type=float, default=0.25)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    scenes = [
    "09c1414f1b",
    "31a2c91c43",
    "1ada7a0617",
    "40aec5fffa",
    "3e8bba0176",
    "e7af285f7d",
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
        tag = f'{scene}_{exp_tag}'
        args.input_path = input_path
        metrics = main(args, exp, tag, scene, args.save_tag, args.transform)
        Log.info(metrics)
        sys.exit(0)

    results = {}
    for scene in tqdm(scenes):
        input_path = '/mnt/bn/haotongdata/Datasets/scannetpp/data/' + scene + '/merge_dslr_iphone'
        exp = args.exp
        exp_tag = args.exp_tag
        tag = f'{scene}_{exp_tag}'
        args.input_path = input_path
        metrics = main(args, exp, tag, scene, args.save_tag, args.transform)
        results[scene] = metrics
    mean = {k:np.mean([results[scene][k] for scene in results]) for k in metrics.keys()} 
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
    