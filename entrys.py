import os
from os.path import join
import argparse
import sys
import ipdb
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from lib.utils.pylogger import Log
import time

# 0706
def evaluate_mesh(args):
    # cmd = 'python3 scripts/scannetpp/fuse_predict_pcd_tsdf_open3d.py --colmap_path colmap/sparse_render_rgb --save_tag no_dist --exp lidar'
    # Log.info('Running: ' + cmd)
    # os.system(cmd)
    # cmd = 'python3 scripts/scannetpp/run_eval.py --exp lidar --save_tag no_dist'
    # Log.info('Running: ' + cmd)
    # os.system(cmd)

    # cmd = 'python3 scripts/scannetpp/fuse_predict_pcd_tsdf_open3d.py --colmap_path colmap/sparse_render_rgb --save_tag no_dist --exp june_depthanythingmetric_scannetpp_0614_hypersim_mask_far'
    # Log.info('Running: ' + cmd)
    # os.system(cmd)
    # cmd = 'python3 scripts/scannetpp/run_eval.py --exp june_depthanythingmetric_scannetpp_0614_hypersim_mask_far --save_tag no_dist'
    # Log.info('Running: ' + cmd)
    # os.system(cmd)

    cmd = 'python3 scripts/scannetpp/fuse_predict_pcd_tsdf_open3d.py --colmap_path colmap/sparse_render_rgb --save_tag no_dist --exp june_depthanythingmetric_scannetpp_0614_hypersim_mask_far_orig_depth_anything_v2'
    Log.info('Running: ' + cmd)
    os.system(cmd)
    cmd = 'python3 scripts/scannetpp/run_eval.py --exp june_depthanythingmetric_scannetpp_0614_hypersim_mask_far_orig_depth_anything_v2 --save_tag no_dist'
    Log.info('Running: ' + cmd)

def fuse_evaluate(args):
    # 生成meshdepth_0614.json
    # cmd = 'python3 scripts/scannetpp/generate_splits_meshdepth_0614.py'
    # os.system(cmd)

    # evalute single mesh
    # cmd = 'python3 main.py exp=depth_estimation/${exp} pl_trainer.devices=8 entry=predict +model.output_tag=${scene}_0707 +data.val_dataset.dataset_opts.scene=${scene} model.save_orig_pred=True data.val_dataset.dataset_opts.frames=\[0,-1,1\]'
    # os.system(cmd)
    # cmd = 'python3 scripts/scannetpp/fuse_predict_pcd_tsdf_open3d.py --colmap_path colmap/sparse_render_rgb --save_tag no_dist --exp ${exp} --exp_tag "0707" --test_one_scene'
    # os.system(cmd)
    # cmd = 'python3 scripts/scannetpp/run_eval.py --test_one_scene --exp_tag "0707" --exp ${exp} --save_tag no_dist'
    # os.system(cmd)


    # predict all depth
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
    exps = [
        'july_hypersim_scannetppmesh',
        'july_hypersim_scannetpp_zip_mesh']
    for scene in scenes:
        for exp in exps:
            cmd = f'python3 main.py exp=depth_estimation/{exp} pl_trainer.devices=8 entry=predict +model.output_tag={scene}_0708_undist_before_input_net +data.val_dataset.dataset_opts.scene={scene} model.save_orig_pred=True data.val_dataset.dataset_opts.frames=\[0,-1,1\]'
            os.system(cmd)
    for exp in exps:
        cmd = f'python3 scripts/scannetpp/fuse_predict_pcd_tsdf_open3d.py --colmap_path colmap/sparse_render_rgb --save_tag no_dist_0708_bn --exp {exp} --exp_tag "0708_undist_before_input_net"'
        os.system(cmd)
        cmd = f'python3 scripts/scannetpp/run_eval.py --exp_tag "0708_undist_before_input_net" --exp {exp} --save_tag no_dist_0708_bn'
        os.system(cmd)

def evaluate_arkit_upsampling(args):
    exp = 'july_hypersim_baseline0717'
    exp_name = 'july_hypersim_baseline0717_4gpus'
    # exp = 'july_hypersim_scannetpp_zip_mesh'
    # exp_name = exp
    # exp = 'july_hypersim_scannetppmesh'
    # exp_name = exp
    exp = 'july_hypersim_baseline0717_minmax'
    exp_name = exp
    exp = 'july_hypersim_baseline0717_crop'
    exp_name = exp
    cmd = f'python3 main.py exp=depth_estimation/{exp} exp_name={exp_name} entry=val pl_trainer.devices=8 +model.output_tag=arkitscenes_subset data=depth_estimation/arkitscenes_subset pl_trainer.limit_val_batches=10000 model.save_vis_depth=True > data/pl_htcode/txts/arkitscenes/upsampling_{exp_name}.txt'
    os.system(cmd)

def evaluate_scannetpp(args):
    # exp = 'july_hypersim_baseline0717'
    # exp_name = 'july_hypersim_baseline0717_4gpus'
    # exp = 'july_hypersim_scannetpp_zip_mesh'
    # exp_name = exp
    # exp = 'july_hypersim_scannetppmesh'
    # exp_name = exp
    exp = 'july_hypersim_baseline0717_minmax'
    exp_name = exp
    exp = 'july_hypersim_baseline0717_crop'
    exp_name = exp
    cmd = f'python3 main.py exp=depth_estimation/{exp} exp_name={exp_name} data=depth_estimation/scannetpp model.save_orig_pred=True +model.output_tag=scannetpp pl_trainer.limit_val_batches=1000 entry=predict +data.val_dataset.dataset_opts.undistort=True'
    os.system(cmd)
    cmd = f'python3 scripts/scannetpp/fuse_predict_pcd_tsdf_open3d.py --exp {exp_name} --save_tag nopost --use_post 0'
    os.system(cmd)

def evaluate_human(args):
    # exp = 'july_hypersim_baseline0717'
    # exp_name = 'july_hypersim_baseline0717_4gpus'
    # exp = 'july_hypersim_scannetpp_zip_mesh'
    # exp_name = exp
    # exp = 'july_hypersim_scannetppmesh'
    # exp_name = exp
    exp = 'july_hypersim_baseline0717_minmax'
    exp_name = exp
    cmd = f'python3 main.py exp=depth_estimation/{exp} exp_name={exp_name} data=depth_estimation/scannetpp model.save_orig_pred=True +model.output_tag=scannetpp pl_trainer.limit_val_batches=1000 entry=predict +data.val_dataset.dataset_opts.undistort=True'
    os.system(cmd)
    cmd = f'python3 scripts/scannetpp/fuse_predict_pcd_tsdf_open3d.py --exp {exp_name} --save_tag nopost --use_post 0'
    os.system(cmd)

def evaluate_scannetpp_colmap_undistort(args):
    # exps = [
    #     'july_hypersim_scannetpp_zip_mesh',
    #     'july_hypersim_scannetppmesh',
    #     'july_hypersim_baseline0717',
    #     'july_hypersim_baseline0717_nograd_both',
    #     'july_hypersim_baseline0717_grad_scannetpp',
    #     'july_hypersim_baseline0717_grad_arkitscenes',
    #     'july_hypersim_baseline0717_minmax',
    #     'july_hypersim_baseline0717_minmax_grad2',
    #     'july_hypersim_baseline0717_minmax_human',
    #     'july_hypersim_baseline0717_minmax_human_all_valid',
    # ]
    # exp_names = [
    #     'july_hypersim_scannetpp_zip_mesh',
    #     'july_hypersim_scannetppmesh',
    #     'july_hypersim_baseline0717_new',
    #     'july_hypersim_baseline0717_nograd_both_new',
    #     'july_hypersim_baseline0717_grad_scannetpp_new',
    #     'july_hypersim_baseline0717_grad_arkitscenes_new',
    #     'july_hypersim_baseline0717_minmax_new_fixbug_1nodes',
    #     'july_hypersim_baseline0717_minmax_grad2',
    #     'july_hypersim_baseline0717_minmax_human',
    #     'july_hypersim_baseline0717_minmax_human_all_valid',
    # ]
    exps = [
        # 'july_hypersim_scannetpp_zip_mesh',
        # 'july_hypersim_scannetppmesh',
        # 'july_hypersim_baseline0717',
        # 'july_hypersim_baseline0717_nograd_both',
        # 'july_hypersim_baseline0717_grad_scannetpp',
        # 'july_hypersim_baseline0717_grad_arkitscenes',
        'july_hypersim_baseline0717_minmax',
        # 'july_hypersim_baseline0717_minmax_grad2',
        'july_hypersim_baseline0717_minmax_human',
        'july_hypersim_baseline0717_minmax_human_all_valid',
        'july_hypersim_baseline0717_minmax_human_all_valid_zip',
    ]
    exp_names = [
        # 'july_hypersim_scannetpp_zip_mesh',
        # 'july_hypersim_scannetppmesh',
        # 'july_hypersim_baseline0717_new',
        # 'july_hypersim_baseline0717_nograd_both_new',
        # 'july_hypersim_baseline0717_grad_scannetpp_new',
        # 'july_hypersim_baseline0717_grad_arkitscenes_new',
        'july_hypersim_baseline0717_minmax_new_fixbug_1nodes',
        # 'july_hypersim_baseline0717_minmax_grad2',
        'july_hypersim_baseline0717_minmax_human',
        'july_hypersim_baseline0717_minmax_human_all_valid',
        'july_hypersim_baseline0717_minmax_human_all_valid_zip',
    ]

    for exp, exp_name in zip(exps, exp_names):
        # cmd = f'python3 main.py exp=depth_estimation/{exp} exp_name={exp_name} data=depth_estimation/scannetpp model.save_orig_pred=True +model.output_tag=scannetpp_colmap_undistort pl_trainer.limit_val_batches=1000 entry=predict +data.val_dataset.dataset_opts.undistort_colmap=True'
        # os.system(cmd)
        cmd = f'python3 scripts/scannetpp/fuse_predict_pcd_tsdf_open3d.py --exp {exp_name} --save_tag colmap_undistort_nopost --use_post 0 --use_colmap_undistort 1'
        os.system(cmd)
        # time.sleep(60)
        # cmd = f'python3 main.py exp=depth_estimation/{exp} exp_name={exp_name} data=depth_estimation/arkitscenes_subset pl_trainer.devices=8 +model.output_tag=arkitscenes_subset pl_trainer.limit_val_batches=1000 entry=val > data/pl_htcode/txts/arkitscenes/upsampling_{exp_name}.txt'
        # os.system(cmd)

def evaluate_scannetpp_arkit(args):
    # exps = [
    #     'july_hypersim_scannetpp_zip_mesh',
    #     'july_hypersim_scannetppmesh',
    #     'july_hypersim_baseline0717',
    #     'july_hypersim_baseline0717_nograd_both',
    #     'july_hypersim_baseline0717_grad_scannetpp',
    #     'july_hypersim_baseline0717_grad_arkitscenes',
    #     'july_hypersim_baseline0717_minmax',
    #     'july_hypersim_baseline0717_minmax_grad2',
    #     'july_hypersim_baseline0717_minmax_human',
    #     'july_hypersim_baseline0717_minmax_human_all_valid',
    # ]
    # exp_names = [
    #     'july_hypersim_scannetpp_zip_mesh',
    #     'july_hypersim_scannetppmesh',
    #     'july_hypersim_baseline0717_new',
    #     'july_hypersim_baseline0717_nograd_both_new',
    #     'july_hypersim_baseline0717_grad_scannetpp_new',
    #     'july_hypersim_baseline0717_grad_arkitscenes_new',
    #     'july_hypersim_baseline0717_minmax_new_fixbug_1nodes',
    #     'july_hypersim_baseline0717_minmax_grad2',
    #     'july_hypersim_baseline0717_minmax_human',
    #     'july_hypersim_baseline0717_minmax_human_all_valid',
    # ]
    exps = [
        # 'july_hypersim_scannetpp_zip_mesh',
        # 'july_hypersim_scannetppmesh',
        # 'july_hypersim_baseline0717',
        # 'july_hypersim_baseline0717_nograd_both',
        # 'july_hypersim_baseline0717_grad_scannetpp',
        # 'july_hypersim_baseline0717_grad_arkitscenes',
        # 'july_hypersim_baseline0717_minmax',
        # 'july_hypersim_baseline0717_minmax_grad2',
        # 'july_hypersim_baseline0717_minmax_human',
        # 'july_hypersim_baseline0717_minmax_human_all_valid',
        # 'july_hypersim_baseline0717_minmax_human_all_valid_zip',
        # 'aug_zipmesh_arkit',
        # 'aug_hypersim_arkit',
        'aug_hypersim_arkit_random',
        'aug_hypersim_arkit_random',
    ]
    exp_names = [
        # 'july_hypersim_scannetpp_zip_mesh',
        # 'july_hypersim_scannetppmesh',
        # 'july_hypersim_baseline0717_new',
        # 'july_hypersim_baseline0717_nograd_both_new',
        # 'july_hypersim_baseline0717_grad_scannetpp_new',
        # 'july_hypersim_baseline0717_grad_arkitscenes_new',
        # 'july_hypersim_baseline0717_minmax_new_fixbug_1nodes',
        # 'july_hypersim_baseline0717_minmax_grad2',
        # 'july_hypersim_baseline0717_minmax_human',
        # 'july_hypersim_baseline0717_minmax_human_all_valid',
        # 'july_hypersim_baseline0717_minmax_human_all_valid_zip',
        # 'repeat_aug_zipmesh_arkit',
        # 'aug_hypersim_arkit',
        'aug_hypersim_arkit_random',
        'aug_hypersim_arkit_random_nospot',
    ]
    
    exps = [# 'aug_hypersim_arkit',
            # 'aug_hypersim_arkit_random',
            # 'aug_hypersim_arkit_random_all_dataset',
            # 'aug_hypersim_arkit_random_all_dataset',
            'aug_hypersim_arkit_random_all_dataset',
            'aug_hypersim_arkit_random_all_dataset',
            ]
    exp_names = [ # 'aug_hypersim_arkit',
                 # 'aug_hypersim_arkit_random',
                 # 'aug_hypersim_arkit_random_all_dataset_grad_grad0.5',
                 # 'aug_hypersim_arkit_random_all_dataset_zip_grad0.5',
                 # 'aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask',
                 'aug_hypersim_arkit_random_all_dataset_zip_human',
                 'aug_hypersim_arkit_random_all_dataset_zip_nohuman',
                 ]
    # exps = ['aug_hypersim_arkit_random_all_dataset']
    # exp_names = ['aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask']
    exps = ['sep13_baseline', 'sep13_baseline']
    exp_names = ['sep13_baseline', 'sep13_baseline_allgrad']
    for exp, exp_name in zip(exps, exp_names):
        cmd = f'python3 main.py exp=depth_estimation/{exp} exp_name={exp_name} data=depth_estimation/scannetpp_new model.save_orig_pred=True +model.output_tag=scannetpp pl_trainer.limit_val_batches=1000 entry=val +data.val_dataset.dataset_opts.undistort=True > data/pl_htcode/txts/scannetpp/upsampling_{exp_name}.txt'
        print(cmd)
        # os.system(cmd)
        cmd = f'python3 scripts/scannetpp/fuse_predict_pcd_tsdf_open3d.py --exp {exp_name} --save_tag post --use_post 1 &'
        # os.system(cmd)
        # time.sleep(60)
        # cmd = f'python3 main.py exp=depth_estimation/{exp} exp_name={exp_name} data=depth_estimation/arkitscenes_subset pl_trainer.devices=8 +model.output_tag=arkitscenes_subset pl_trainer.limit_val_batches=1000 entry=val > data/pl_htcode/txts/arkitscenes/upsampling_{exp_name}.txt'
        # os.system(cmd)

def main(args):
    '''python3 scripts/mycapture/mege_plot.py'''
    input1s = [
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split10/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split15/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split25/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split50/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split250/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split10/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split15/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split25/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split50/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split250/traj/ours_30000/renders',
    ]

    input2s = [
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_lidar_l10_split10/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_lidar_l10_split15/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_lidar_l10_split25/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_lidar_l10_split50/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_lidar_l10_split250/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_rgb_correct_init_split10/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_rgb_correct_init_split15/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_rgb_correct_init_split25/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_rgb_correct_init_split50/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_rgb_correct_init_split250/traj/ours_30000/renders',
    ]

    name1s = ['ours'] * 10
    name2s = ['lidar'] * 5 + ['rgb'] * 5
    outputs = [
        'ours_lidar_comp_{}views'.format(10),
        'ours_lidar_comp_{}views'.format(15),
        'ours_lidar_comp_{}views'.format(25),
        'ours_lidar_comp_{}views'.format(50),
        'ours_lidar_comp_{}views'.format(250),
        'ours_rgb_comp_{}views'.format(10),
        'ours_rgb_comp_{}views'.format(15),
        'ours_rgb_comp_{}views'.format(25),
        'ours_rgb_comp_{}views'.format(50),
        'ours_rgb_comp_{}views'.format(250),
    ]
    outputs = ['data/pl_htcode/demos/{}.mp4'.format(item) for item in outputs]

    input1s = [
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split7/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split7/traj/ours_30000/renders',
    ]

    input2s = [
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_lidar_l10_split7/traj/ours_30000/renders',
        '/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_rgb_correct_init_split7/traj/ours_30000/renders',
    ]

    name1s = ['ours'] * 2
    name2s = ['lidar']  + ['rgb'] 
    outputs = [
        'ours_lidar_comp_{}views'.format(7),
        'ours_rgb_comp_{}views'.format(7),
    ]
    outputs = ['data/pl_htcode/demos/{}.mp4'.format(item) for item in outputs]
    for input1, input2, name1, name2, output in zip(input1s, input2s, name1s, name2s, outputs):
        cmd = 'python3 scripts/mycapture/mege_plot.py --input1 {} --input2 {} --name1 {} --name2 {} --output {}'.format(input1, input2, name1, name2, output)
        os.system(cmd)

def run_mycapture(args):
    input_data_dir = '/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner'
    scene = '3210c6facd' # d1bb5a1af4
    rot = 0
    cmd = f'python3 scripts/mycapture/run.py --input_data_dir {input_data_dir} --scene {scene} --rot {rot}'
    os.system(cmd)
    cmd = f'python3 scripts/mycapture/run_spsg.py --input_data_dir {input_data_dir} --scene {scene}'
    os.system(cmd)
    cmd = f'python3 scripts/mycapture/run_compute_colmap_scale.py --input_path {input_data_dir} --scene {scene}'
    os.system(cmd)
    
    # colmap image_undistorter --input_path $PWD/85154946bc/colmap/sparse/0_metric --output_path $PWD/85154946bc/dense --image_path $PWD/85154946bc/images
    cmd = f'colmap image_undistorter --input_path {input_data_dir}/{scene}/colmap/sparse/0_metric --output_path {input_data_dir}/{scene}/dense --image_path {input_data_dir}/{scene}/images'
    os.system(cmd)
    cmd = f'ln -s {input_data_dir}/{scene}/dense {input_data_dir}/{scene}/colmap/dense'
    os.system(cmd)

def run_debug_scannetpp_scale(args):
    scene = '578511c8a9' # 1.0070798107225873
    scene = '5f99900f09' # 0.9987619658377518
    from lib.utils.colmap_scale import compute_scale

    colmap_path = '/mnt/bn/haotongdata/Datasets/scannetpp/data/{}/merge_dslr_iphone/colmap/sparse_render_rgb'.format(scene)
    depth_path = '/mnt/bn/haotongdata/Datasets/scannetpp/data/{}/iphone/depth'.format(scene)
    a = compute_scale(colmap_path, 
                      depth_path)
    
def run_evaluate_human(args):
    from lib.utils import vis_utils
    # 研究simulate方式
    # 研究真实数据
    exps = ['aug_hypersim_arkit',
            'aug_hypersim_arkit_random',
            'aug_hypersim_arkit_random_all_dataset',
            'aug_hypersim_arkit_random_all_dataset',
            'aug_hypersim_arkit_random_all_dataset',
            'aug_hypersim_arkit_random_all_dataset',
            'aug_hypersim_arkit_random_all_dataset',
            'aug_hypersim_arkit_random_all_dataset',
            ]
    exp_names = ['aug_hypersim_arkit',
                 'aug_hypersim_arkit_random',
                 'aug_hypersim_arkit_random_all_dataset_grad_grad0.1',
                 'aug_hypersim_arkit_random_all_dataset_grad_grad0.5',
                 'aug_hypersim_arkit_random_all_dataset_grad_grad2',
                 'aug_hypersim_arkit_random_all_dataset_zip_grad0.1',
                 'aug_hypersim_arkit_random_all_dataset_zip_grad0.5',
                 'aug_hypersim_arkit_random_all_dataset_zip_grad2',
                 ]
    for exp, exp_name in zip(exps, exp_names):
        cmd = f'python3 main.py exp=depth_estimation/{exp} exp_name={exp_name} data=depth_estimation/mycapture_v3_dyn entry=predict data.val_dataset.dataset_opts.frames=\[0,600,1\] +model.output_tag=dyn_20240727_seq1 model.save_orig_pred=True +model.near_depth=0.5 +model.far_depth=6.5'
        os.system(cmd)
        # read_depth_func = lambda x: np.asarray(imageio.imread(x))/1000
        cam = '30'
        rgb_dir = os.path.join('/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/20240728_seq1/images_undist', cam)
        depth_dir = f'/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/{exp_name}/results/dyn_20240727_seq1/orig_pred'
        tar_dir = f'/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/{exp_name}/results/dyn_20240727_seq1/{cam}'
        vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 300, 1], fps=15, depth_format='.npz', depth_prefix=f'{cam}_')
        
        
        cam = '31'
        rgb_dir = os.path.join('/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/20240728_seq1/images_undist', cam)
        depth_dir = f'/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/{exp_name}/results/dyn_20240727_seq1/orig_pred'
        tar_dir = f'/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/{exp_name}/results/dyn_20240727_seq1/{cam}'
        vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 300, 1], fps=15, depth_format='.npz', depth_prefix=f'{cam}_')

        cam = '33'
        rgb_dir = os.path.join('/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/20240728_seq1/images_undist', cam)
        depth_dir = f'/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/{exp_name}/results/dyn_20240727_seq1/orig_pred'
        tar_dir = f'/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/{exp_name}/results/dyn_20240727_seq1/{cam}'
        vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 300, 1], fps=15, depth_format='.npz', depth_prefix=f'{cam}_')


def run_evaluate_waymo(args):
    exps = [
        'aug_da_log_fix_baseline',
        'aug_da_log_fix_baseline_resize',
        'aug_da_log_fix_baseline_resize_sq',
        'aug_hypersim_kitti',
        'aug_shift_comp_baseline',
        'aug_shift_comp_baseline_new_lidar',
        'aug_shift_comp_baseline_new_lidar_precomp',
        'aug_shift_comp_baseline_new_lidar_precomp',
        'aug_shift_comp_baseline_new_lidar_warpminmax',
        'aug_shift_comp_baseline_new_lidar_warpminmax_precomp',
        'aug_shift_comp_baseline_new_lidar_warpminmax_precomp',
    ]
    exp_names = [
        'aug_da_log_fix_baseline',
        'aug_da_log_fix_baseline_resize',
        'aug_da_log_fix_baseline_resize_sq',
        'aug_hypersim_kitti',
        'aug_shift_comp_baseline',
        'aug_shift_comp_baseline_new_lidar',
        'aug_shift_comp_baseline_new_lidar_precomp',
        'aug_shift_comp_baseline_new_lidar_precomp_nograd',
        'aug_shift_comp_baseline_new_lidar_warpminmax',
        'aug_shift_comp_baseline_new_lidar_warpminmax_precomp',
        'aug_shift_comp_baseline_new_lidar_warpminmax_precomp_nograd',
    ]
    os.makedirs('data/pl_htcode/txts/waymo', exist_ok=True)
    for exp, exp_name in zip(exps, exp_names):
        try:
            cmd = f'python3 main.py exp=depth_estimation/{exp} data=depth_estimation/waymo_comp +model.output_tag=waymo_002 entry=val model.save_orig_pred=True exp_name={exp_name} > data/pl_htcode/txts/waymo/metric_002_{exp_name}.txt'
            os.system(cmd)
            Log.info('Running: ' + cmd)
        except:
            Log.error('Error in {}'.format(exp_name))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='help')
    args = parser.parse_args()
    return args


def help():
    print('evaluation depth command:')
    'export exp=aug_hypersim_arkit_random_all_dataset'

if __name__ == '__main__':
    args = parse_args()
    globals()[args.func](args)