export workspace=/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode

export scene=ai_003_007
export cam=cam_00
mlx worker launch --cpu 11 --gpu 1 python3 main.py exp=mde/mde_upsample_baseline_0404_aug_simu_linear_byte entry=val data=mde/hypersim_val_byte +model.args.save_vis_pred=True +model.args.save_orig_pred=True +model.args.save_align_pred=True save_tag=${scene}-${cam} data.dataset_opts.val.scene=${scene} data.dataset_opts.val.cam=${cam}

export scene=ai_003_007
export cam=cam_01
mlx worker launch --cpu 11 --gpu 1 python3 main.py exp=mde/mde_upsample_baseline_0404_aug_simu_linear_byte entry=val data=mde/hypersim_val_byte +model.args.save_vis_pred=True +model.args.save_orig_pred=True +model.args.save_align_pred=True save_tag=${scene}-${cam} data.dataset_opts.val.scene=${scene} data.dataset_opts.val.cam=${cam}

export scene=ai_005_001 # failed
export cam=cam_00
mlx worker launch --cpu 11 --gpu 1 python3 main.py exp=mde/mde_upsample_baseline_0404_aug_simu_linear_byte entry=val data=mde/hypersim_val_byte +model.args.save_vis_pred=True +model.args.save_orig_pred=True +model.args.save_align_pred=True save_tag=${scene}-${cam} data.dataset_opts.val.scene=${scene} data.dataset_opts.val.cam=${cam}

export scene=ai_005_010
export cam=cam_00
mlx worker launch --cpu 11 --gpu 1 python3 main.py exp=mde/mde_upsample_baseline_0404_aug_simu_linear_byte entry=val data=mde/hypersim_val_byte +model.args.save_vis_pred=True +model.args.save_orig_pred=True +model.args.save_align_pred=True save_tag=${scene}-${cam} data.dataset_opts.val.scene=${scene} data.dataset_opts.val.cam=${cam}

export scene=ai_008_005
export cam=cam_00
mlx worker launch --cpu 11 --gpu 1 python3 main.py exp=mde/mde_upsample_baseline_0404_aug_simu_linear_byte entry=val data=mde/hypersim_val_byte +model.args.save_vis_pred=True +model.args.save_orig_pred=True +model.args.save_align_pred=True save_tag=${scene}-${cam} data.dataset_opts.val.scene=${scene} data.dataset_opts.val.cam=${cam}

export scene=ai_008_007
export cam=cam_00
mlx worker launch --cpu 11 --gpu 1 python3 main.py exp=mde/mde_upsample_baseline_0404_aug_simu_linear_byte entry=val data=mde/hypersim_val_byte +model.args.save_vis_pred=True +model.args.save_orig_pred=True +model.args.save_align_pred=True save_tag=${scene}-${cam} data.dataset_opts.val.scene=${scene} data.dataset_opts.val.cam=${cam}

export scene=ai_010_004
export cam=cam_00
mlx worker launch --cpu 11 --gpu 1 python3 main.py exp=mde/mde_upsample_baseline_0404_aug_simu_linear_byte entry=val data=mde/hypersim_val_byte +model.args.save_vis_pred=True +model.args.save_orig_pred=True +model.args.save_align_pred=True save_tag=${scene}-${cam} data.dataset_opts.val.scene=${scene} data.dataset_opts.val.cam=${cam}


export scene=ai_011_007
export cam=cam_00
mlx worker launch --cpu 11 --gpu 1 python3 main.py exp=mde/mde_upsample_baseline_0404_aug_simu_linear_byte entry=val data=mde/hypersim_val_byte +model.args.save_vis_pred=True +model.args.save_orig_pred=True +model.args.save_align_pred=True save_tag=${scene}-${cam} data.dataset_opts.val.scene=${scene} data.dataset_opts.val.cam=${cam}

