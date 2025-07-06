# -- coding: utf-8 --
# @Time : 2023/2/7 15:04
# @Author : LZL
# @Email: z.luan@qmul.ac.uk
# @File : generate_7Scenes_pcd.py

import os
import os.path as osp
import numpy as np
import cv2

'''
convert caliberated 2D depth image to 3D point cloud
'''


def get_depth(depth, calibration_extrinsics, intrinsics_color, intrinsics_depth_inv):
    """Return the calibrated depth image (7-Scenes).
    Calibration parameters from DSAC (https://github.com/cvlab-dresden/DSAC) used.
    This code is transforming the depth image from one coordinate system to another, based on extrinsic and intrinsic camera parameters.
    The intrinsic parameters describe the internal parameters of the camera lens and sensor, and the extrinsic parameters describe the position
    of the camera in the world.
    """
    img_height, img_width = depth.shape[0], depth.shape[1]
    depth_ = np.zeros_like(depth)
    x = np.linspace(0, img_width - 1, img_width)
    y = np.linspace(0, img_height - 1, img_height)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, (1, -1))
    yy = np.reshape(yy, (1, -1))
    ones = np.ones_like(xx)
    pcoord_depth = np.concatenate((xx, yy, ones), axis=0)
    depth = np.reshape(depth, (1, img_height * img_width))

    # Transfer the 2D image(depth) coordinates (pcoord_depth) into
    # 3D camera coordinates (ccoord_depth) in the camera's own coordinate system
    # By multiplying this 3D point with the depth value, we scale the point such that
    # its z-coordinate (the coordinate that corresponds to the depth value) matches the
    # actual distance from the camera to the point. This gives us the actual 3D position of the point in the depth camera coordinate system.
    ccoord_depth = np.dot(intrinsics_depth_inv, pcoord_depth) * depth

    ccoord_depth[1, :] = - ccoord_depth[1, :]  # y->-y, convert OpenCV Coordinate to OpenGL Coordinate
    ccoord_depth[2, :] = - ccoord_depth[2, :]  # z->-z, convert OpenCV Coordinate to OpenGL Coordinate
    ccoord_depth = np.concatenate((ccoord_depth, ones), axis=0)
    ccoord_color = np.dot(calibration_extrinsics, ccoord_depth)  # align depth and image, multiply by Transfer Matrix
    ccoord_color = ccoord_color[0:3, :]

    # To align the depth coordinate system used in the depth image with the camera coordinate system used in the color image.
    # This type of coordinate system conversion or coordinate system alignment is often necessary when working
    # with multiple sensors or cameras that use different coordinate systems.
    ccoord_color[1, :] = - ccoord_color[1, :]
    ccoord_color[2, :] = depth

    pcoord_color = np.dot(intrinsics_color, ccoord_color)  # gives the location of the 3D points in the image plane of the camera （p=KP）
    pcoord_color = pcoord_color[:, pcoord_color[2, :] != 0]

    # pcoord_color[0,:] represents the x-coordinate of the projected 3D point in the color image plane.
    # The line pcoord_color[0,:] = pcoord_color[0,:]/pcoord_color[2,:]+0.5 is normalizing the x-coordinate by dividing it
    # by the corresponding depth value pcoord_color[2,:] and adding 0.5 to the result. This is done to convert the x-coordinate
    # from homogeneous to cartesian coordinate system and to make it zero-index based. Adding 0.5 to the result is equivalent
    # to rounding up to the nearest integer, which makes it an index value.
    # This operation is performed to map the 3D point to a specific pixel in the color image.
    # Note: thislike [x/z,y/z,z] to convert the x-coordinate from homogeneous to cartesian coordinate system, because here:
    #  ccoord_depth = np.dot(intrinsics_depth_inv, pcoord_depth) * depth, the 'x' and 'y' coordinate is multiply by 'depth'.
    pcoord_color[0, :] = pcoord_color[0, :] / pcoord_color[2, :] + 0.5
    pcoord_color[0, :] = pcoord_color[0, :].astype(int)
    pcoord_color[1, :] = pcoord_color[1, :] / pcoord_color[2, :] + 0.5
    pcoord_color[1, :] = pcoord_color[1, :].astype(int)
    pcoord_color = pcoord_color[:, pcoord_color[0, :] >= 0]
    pcoord_color = pcoord_color[:, pcoord_color[1, :] >= 0]
    pcoord_color = pcoord_color[:, pcoord_color[0, :] < img_width]
    pcoord_color = pcoord_color[:, pcoord_color[1, :] < img_height]

    depth_[pcoord_color[1, :].astype(int), pcoord_color[0, :].astype(int)] = pcoord_color[2, :]

    # The final coordinate of depth_ is camera color image coordinate. It maps the depth information back to the camera color image,
    # meaning each pixel in depth_ represents the depth information at the corresponding position in the camera color image.
    return depth_


def generate_pcd(depth_file, save_filename):
    # configs for calibrated depth (https://github.com/AaltoVision/hscnet/blob/master/datasets/seven_scenes.py)
    intrinsics_color = np.array([[525.0, 0.0, 320.0],
                                 [0.0, 525.0, 240.0],
                                 [0.0, 0.0, 1.0]])

    intrinsics_depth = np.array([[585.0, 0.0, 320.0],
                                 [0.0, 585.0, 240.0],
                                 [0.0, 0.0, 1.0]])

    intrinsics_depth_inv = np.linalg.inv(intrinsics_depth)
    calibration_extrinsics = np.loadtxt(os.path.join('/public/home/luanzl/WorkSpace/DFNet', 'data', '7Scenes', 'sensorTrans.txt'))

    depth = cv2.imread(depth_file, -1).astype(np.float32)  # depth is saved 16-bit PNG in millimeters
    depth[depth >= 65535] = 0
    depth_ = get_depth(depth, calibration_extrinsics, intrinsics_color, intrinsics_depth_inv)

    # Get image dimensions
    height, width = depth_.shape[0], depth_.shape[1]

    # Create a blank array to store the point cloud data
    points = np.zeros((height * width, 3), np.float32)

    # Populate the points array with x, y, and z data
    for i in range(height):
        for j in range(width):
            z = depth_[i, j]
            if z > 0:
                x = (j - width / 2) * z / 585.
                y = (i - height / 2) * z / 585.
                points[i * height + j] = [x, y, z]

    # save point cloud data
    points.tofile(save_filename)
    print('Save [%s]' % save_filename)


if __name__ == '__main__':
    # directories
    scenes = ['fire', 'office', 'pumpkin', 'redkitchen', 'stairs']
    print(scenes)

    data_path = r'/public/home/luanzl/WorkSpace/DFNet/data/deepslam_data'

    for scene in scenes:
        data_dir = osp.join(data_path, '7Scenes', scene)

        split_file = osp.join(data_dir, 'TrainSplit.txt')
        with open(split_file, 'r') as f:
            train_seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

        split_file = osp.join(data_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            test_seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

        seqs = train_seqs + test_seqs

        tmp_depths = []
        for seq in seqs:
            seq_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir)) if n.find('pose') >= 0]
            frame_idx = np.array(range(len(p_filenames)))
            depths = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i)) for i in frame_idx]
            tmp_depths.extend(depths)

        for tmp_depth in tmp_depths:
            index = tmp_depth.find('.')
            path = tmp_depth[:index]
            generate_pcd(depth_file=tmp_depth, save_filename='%s.bin' % path)

        print('Processed scene: %s' % str(scene))

    print('Processed all files!')
