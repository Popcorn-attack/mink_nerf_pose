# -- coding: utf-8 --
# @Time : 2023/2/8 17:07
# @Author : LZL
# @Email: z.luan@qmul.ac.uk
# @File : project_pcd_to_image.py

'''
project pcd (convert from calibrated depth image) to color RGB image
'''

import numpy as np
import cv2

# Load the point cloud bin file
point_cloud = np.fromfile(r'/Users/williamed/Desktop/WorkSpace/temp/frame-000000.bin', dtype=np.float32).reshape(-1, 3)

# Filter the point cloud to only keep points with z-value greater than zero
point_cloud = point_cloud[point_cloud[:, 2] > 0]

# Camera intrinsic parameters (Note: use depth intrinsic)
fx, fy, cx, cy = 585, 585, 320, 240

# Project the point cloud to 2D pixels
projected_points = np.zeros((point_cloud.shape[0], 2), dtype=np.int32)
projected_points[:, 0] = np.round(fx * point_cloud[:, 0] / point_cloud[:, 2] + cx).astype(np.int32)
projected_points[:, 1] = np.round(fy * point_cloud[:, 1] / point_cloud[:, 2] + cy).astype(np.int32)

# Load the RGB image
img = cv2.imread(r'/Users/williamed/Desktop/WorkSpace/temp/frame-000000.color.png')

# Plot the projected point cloud on the RGB image
for x, y in projected_points:
    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

cv2.imshow('Point Cloud on RGB Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
