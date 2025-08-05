import numpy as np
import os 
from os.path import join

data_path = '/mnt/bn/haotongdata/home/linhaotong/projects/IGEV/IGEV-MVS/DTU/Cameras'
# 6views
views = [25, 22, 28, 40, 44, 48]
# 9views
# views = [25, 22, 28, 40, 44, 48, 0, 8, 13]
# 14views
# views = [13, 16, 22, 23, 24, 25, 28, 27, 29, 31, 34, 35, 47, 48]
print(len(views), ' : ', views)
def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[-1])
    depth_max = 925
    return intrinsics, extrinsics, depth_min, depth_max
pos = []
for view in views:
    view_path = join(data_path, '{:08d}_cam.txt'.format(view))
    _, ext, _, _ = read_cam_file(view_path)
    c2w = np.linalg.inv(ext)
    pos.append(c2w[:3, 3])
pos = np.array(pos)
distance_matrix = np.linalg.norm(pos[:, None] - pos[None], axis=-1)



for idx, view in enumerate(views):
    ids = distance_matrix[idx].argsort()
    print('metas.append((scan, {}, [{}, {}, {}, {}]))'.format(view, views[ids[1]], views[ids[2]], views[ids[3]], views[ids[4]]))
  
print('') 
    
for idx, view in enumerate(views):
    ids = distance_matrix[idx].argsort()
    if idx == 0:
        print('pair_data = [({}, [{}, {}, {}, {}])]'.format(view, views[ids[1]], views[ids[2]], views[ids[3]], views[ids[4]]))
    else:
        print('pair_data += [({}, [{}, {}, {}, {}])]'.format(view, views[ids[1]], views[ids[2]], views[ids[3]], views[ids[4]]))