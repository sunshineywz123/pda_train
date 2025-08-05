import hydra
from lib.entrys import *
import os
import socket

from lib.utils.pylogger import Log

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg) -> None:
    globals()[cfg.entry](cfg)
    
def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return False  # 如果能绑定，则端口未被占用
        except socket.error:
            return True   # 如果发生错误，意味着端口已被占用

if __name__ == "__main__":
    assert 'workspace' in os.environ
    if 'HTCODE_DEBUG_DDP' in os.environ:
        Log.info('DEBUG DDP')
    elif 'ARNOLD_WORKER_0_HOST' in os.environ:
        os.environ['MASTER_IP'] = os.environ['ARNOLD_WORKER_0_HOST']
        os.environ['MASTER_ADDR'] = os.environ['ARNOLD_WORKER_0_HOST']
        if check_port(int(os.environ['ARNOLD_WORKER_0_PORT'])) == False:
            os.environ['MASTER_PORT'] = os.environ['ARNOLD_WORKER_0_PORT']
        os.environ['NODE_SIZE'] = os.environ['ARNOLD_WORKER_NUM']
        os.environ['NODE_RANK'] = os.environ['ARNOLD_ID']
    main()
 