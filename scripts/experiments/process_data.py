import sys
sys.path.append('.')
from lib.utils.parallel_utils import parallel_execution

scenes = [
    "20240522_bicycle",
    "20240629_centerflower",
    "20240629_statue1",
    "20240702_statue1",
    "20240702_statue2",
    "20240702_statue3",
    "20240702_statue4",
    "20240702_statue5",
]

def process_scene(scene):

    cmd = f
    # 提取depth
    # 跑colmap
    # 对齐depth和colmap
    pass


parallel_execution(
    scenes,
    action=process_scene
)




