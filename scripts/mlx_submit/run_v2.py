from lib.utils.pylogger import Log
from datetime import datetime
import yaml
import os
from os.path import join
import argparse
from pydoc import describe
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import subprocess

def get_git_info():
    branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('utf-8').strip()
    commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('utf-8').strip()[:6]
    return branch_name, commit_id

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str,
                        default='/mlx_devbox/users/linhaotong/repo/9203/pl_htcode')
    parser.add_argument('--default_yaml', type=str,
                        default='/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/configs/mlx_configs/default_v2.yaml')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--job_name', type=str)
    parser.add_argument('--default_output_dir', type=str,
                        default='/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/configs/mlx_configs/pl_htcode')

    parser.add_argument('--num_cpus', type=int, default=11,
                        help='num of cpus per gpu')
    parser.add_argument('--num_mem', type=int,
                        default=40, help='MB per gpu')
    parser.add_argument('--default_prefix_command', type=str,
                        default='/bin/bash /opt/tiger/debug/launch.sh ')
                        # default='/bin/bash /opt/tiger/debug/launch.sh MASTER_IP=$ARNOLD_WORKER_0_HOST MASTER_ADDR=$ARNOLD_WORKER_0_HOST MASTER_PORT=$ARNOLD_WORKER_0_PORT NODE_SIZE=$ARNOLD_WORKER_NUM NODE_RANK=$ARNOLD_ID ')
                        # default='/bin/bash /mnt/bn/haotongdata/home/linhaotong/envs/debug/launch.sh ')

    parser.add_argument('--desc', type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def main(args):
    # 创建yaml
    # 1. load default yaml
    # 2. 根据gpu数量和args， 修改yaml
    # 3.
    mlx_config = yaml.load(open(args.default_yaml, 'r'),
                           Loader=yaml.FullLoader)
    # if args.num_nodes > 1: raise NotImplementedError

    mlx_config['jobDefVersion']['resource']['arnoldConfig']['roles'][0]['cpu'] = args.num_gpus * args.num_cpus
    mlx_config['jobDefVersion']['resource']['arnoldConfig']['roles'][0]['gpu'] = args.num_gpus
    mlx_config['jobDefVersion']['resource']['arnoldConfig']['roles'][0]['memory'] = args.num_gpus * args.num_mem * 1024
    mlx_config['jobDefVersion']['resource']['arnoldConfig']['roles'][0]['num'] = args.num_nodes
    

    # 把remainder args作为command参数加到mlx_config['arnold']['entrypoint']中
    command = args.default_prefix_command + ' '.join(args.opts)
    mlx_config['jobDefVersion']['entrypointFullScript'] = command
    mlx_config['jobRunParams']['entrypointFullScript'] = command
    

    job_name = args.job_name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    branch, commit_id = get_git_info()
    # job_name = timestamp + '_' + job_name
    job_name = f'{timestamp}_{job_name}_{branch}_{commit_id}'
    output_path = join(args.default_output_dir, job_name + '.yaml')
    yaml.dump(mlx_config, open(output_path, 'w'))
    Log.info(f'Yaml file saved to {output_path}')

    try:
        pwd = os.getcwd()
        os.chdir(args.workspace)
        os.system(f'mlx job submitv2 --caption {job_name} --path {output_path}')
        Log.info('submit job success')
        os.chdir(pwd)
    except:
        Log.error('submit job failed')
        Log.error('clean yaml file')
        os.remove(output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)


'''
把下面的文件加入到.zshrc或.bashrc中
mlx_submit_v2() {
  if [[ -e scripts/mlx_submit/run_v2.py ]]; then
    python3 scripts/mlx_submit/run_v2.py "$@"
  else
    echo "错误：scripts/mlx_submit/run.py 不存在，请确认路径是否正确。"
  fi
}
'''
