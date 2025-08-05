## 02.13
检测目前的代码有没有问题，因为hypersim train出来似乎不稳定了，看看是不是vkitti的影响。
```
python main.py exp=marigold/marigold_disp_only_hypersim data.loader_opts.train.batch_size=2 pl_trainer.resume_training=False

python main.py exp=marigold/marigold_disp data.loader_opts.train.batch_size=2 pl_trainer.accumulate_grad_batches=16 resume_training=False pl_trainer.devices=\[1\]


ant_submit --job_cfg configs/local/job_config_8gpus.yml --command HYDRA_FULL_ERROR=1 python main.py exp=marigold/marigold_disp_only_hypersim exp_name=marigold_disp_only_hypersim_debug pl_trainer.devices=8

ant_submit --job_cfg configs/local/job_config_8gpus.yml --command HYDRA_FULL_ERROR=1 python main.py exp=marigold/marigold_disp exp_name=marigold_disp_debug pl_trainer.devices=8 +pl_trainer.use_distributed_sampler=False

ant_submit --job_cfg configs/local/job_config_32gpus.yml --command HYDRA_FULL_ERROR=1 python main.py exp=marigold/marigold_disp_only_hypersim exp_name=marigold_disp_only_hypersim_debug_32gpus pl_trainer.devices=8 +pl_trainer.num_nodes=4
```

```
 python main.py exp=dpt/dpt pl_trainer.accumulate_grad_batches=16 data.loader_opts.train.batch_size=6
```

## 2.16

```
bea804: python main.py exp=dpt/dpt data.loader_opts.train.batch_size=6 exp_name=dpt_debug_grad1
# 检验了accumelate batches的影响
```

```
98f944746e: python main.py exp=dpt/dpt data.loader_opts.train.batch_size=6 exp_name=dpt_debug_lr pl_trainer.devices=\[1\] pl_trainer.accumulate_grad_batches=16 
# 检验了learning rate和pretrain model, 1e-5 for backbone, 1e-4 for head
```

## 2.17 

```
# 检验 finetune model, and scale invariant loss
python main.py exp=dpt/dpt exp_name=dpt_debug_ftssi pl_trainer.accumulate_grad_batches=16 data=dpt/hypersim_nonorm data.loader_opts.train.batch_size=4 
# 检验 finetune model, and scale invartiance loss in disparity space
invert = False
python main.py exp=dpt/dpt exp_name=dpt_debug_ftssi_inv pl_trainer.accumulate_grad_batches=16 data=dpt/hypersim_disp data.loader_opts.train.batch_size=4
```

## 2.19

```
python main.py exp=dpt/dpt_orig print_cfg=True exp_name=debug_zoeloss +model.pipeline.args.loss=zoe pl_trainer.accumulate_grad_batches=8 resume_training=False
```