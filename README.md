# pytorch-lighting htcode

Minimal marigold training and inference with multiple devices and machines

## 0. 环境准备
### 0.1 python环境
`
pip install -r requirements.txt
`
即可

注意pytorch >= 2.0; pytorch_lightning >= 2.0；以及cuda >= 11.7

### 0.2 数据准备
下载hypersim数据集
```
ROOT=PATH_HYPERSIM_ROOT
mkdir -p $ROOT/data
cd $ROOT
git clone git@github.com:apple/ml-hypersim.git
cd ml-hypersim/contrib/99991
./download.py --contains ai_001_001 --contains scene_cam_00_final_preview --contains tonemap.jpg --silent -d $ROOT/data
./download.py --contains ai_001_001 --contains scene_cam_00_geometry_hdf5 --contains depth_meters.hdf5 --silent -d $ROOT/data
```

这里的depth_meters.hdf5是ray_distance，如果你需要训练一个真正的depth模型，需要根据内参把它们转成真正的depth。

如果想下载足够多的数据，把--contains ai_001_001 --contains scene_cam_00_final_preview 去掉即可。

把数据连接到$workspace，

```
mkdir -p $workspace/datasets
cd $workspace/datasets
ln -s $ROOT/data hypersim_mini
```
workspace的设计初衷是为了不同机器（集群/本地）环境的迁移成本，在各个类型的环境下，setup一个相同的工作目录即可。workspace一般用于存储训练/测试过程中产生的中间结果，最好放在一个比较大的硬盘上。


预处理data
```
python3 scripts/mini_marigold/preprocess_hypersim.py --input $workspace/datasets/hypersim_mini --output $workspace/processed_datasets/hypersim_mini/metadata.json
```
### 0.3 预训练模型准备

```
# marigold checkpoint
mkdir -p $workspace/cache_models
cd $workspace/cache_models
git clone git@github.com:prs-eth/Marigold.git
cd Marigold
bash script/download_weights.sh marigold-v1-0

# stable diffusion checkpoint
sudo apt install git-lfs
git lfs install
cd $workspace/cache_models
git clone https://huggingface.co/stabilityai/stable-diffusion-2
```



## 1. 测试 official marigold

```
python3 main.py exp=depth_estimation/${exp} exp_name=${exp}_official entry=val
```

## 2. 训练 marigold, 单机多卡

```
python3 main.py exp=depth_estimation/${exp} exp_name=${exp} pl_trainer.devices=8 callbacks.model_checkpoint.every_n_epochs=50
```

## 3. 训练marigold, 测试多机多卡
在一台机器上测试多机多卡
```
HTCODE_DEBUG_DDP=1 CUDA_VISIBLE_DEVICES=0, NODE_SIZE=2 NODE_RANK=0 MASTER_IP=localhost MASTER_ADDR=localhost MASTER_PORT=9999 python3 main.py exp=depth_estimation/${exp} exp_name=${exp} pl_trainer.devices=1 pl_trainer.num_nodes=2
HTCODE_DEBUG_DDP=1 CUDA_VISIBLE_DEVICES=1, NODE_SIZE=2 NODE_RANK=1 MASTER_IP=localhost MASTER_ADDR=localhost MASTER_PORT=9999 python3 main.py exp=depth_estimation/${exp} exp_name=${exp} pl_trainer.devices=1 pl_trainer.num_nodes=2
```



## FAQ

### 1. 学习cfg 
```
export HYDRA_FULL_ERROR=1 
export exp=mini_marigold
python3 main.py exp=depth_estimation/${exp} exp_name=debug_${exp} entry=debug_cfg

python3 main.py exp=depth_estimation/${exp} exp_name=debug_${exp} entry=debug_cfg pl_trainer.devices=8

python3 main.py exp=depth_estimation/${exp} exp_name=debug_${exp} entry=debug_cfg +data.train_loader_opts.num_workers=0
# 如果原config没有的话，需要添加+
```

### 2. debug dataloader

```
python3 main.py exp=depth_estimation/${exp} exp_name=debug_${exp} entry=debug_train_dataloader
python3 main.py exp=depth_estimation/${exp} exp_name=debug_${exp} entry=debug_val_dataloader
```

