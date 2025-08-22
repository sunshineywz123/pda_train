import cv2
import numpy as np
from PIL import Image

# 读取 PNG 图像
img = Image.open("/iag_ad_01/ad/yuanweizhong/datasets/shift/0a03-8855/00000000_depth_front.png")
img = np.array(img).astype(np.uint32)

# 解码为整数深度值
# depth_raw = img[:, :, 0] + img[:, :, 1] * 256 + img[:, :, 2] * 256 * 256

depth_raw = img

# 转换为米
depth_meters = depth_raw.astype(np.float32) 

# 可视化保存（仅用于显示）
depth_vis = cv2.normalize(depth_meters, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("decoded_depth_in_meters.png", depth_vis)
