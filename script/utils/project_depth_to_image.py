import numpy as np
import cv2

'''
project caliberated depth image to color RGB image
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
    # 3D camera coordinates (ccoord_depth) in the camera's own coordinate system.
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


# Load the RGB image
img_file = r'/Users/williamed/Desktop/WorkSpace/DFNet/data/deepslam_data/tmp/seq-01/frame-000000.color.png'
img = cv2.imread(img_file)

# Load the depth png file
depth_file = r'/Users/williamed/Desktop/WorkSpace/DFNet/data/deepslam_data/tmp/seq-01/frame-000000.depth.png'
intrinsics_color = np.array([[525.0, 0.0, 320.0],
                             [0.0, 525.0, 240.0],
                             [0.0, 0.0, 1.0]])

intrinsics_depth = np.array([[585.0, 0.0, 320.0],
                             [0.0, 585.0, 240.0],
                             [0.0, 0.0, 1.0]])

intrinsics_depth_inv = np.linalg.inv(intrinsics_depth)
calibration_extrinsics = np.loadtxt(r'/Users/williamed/Desktop/WorkSpace/DFNet/data/deepslam_data/tmp/sensorTrans.txt')

depth = cv2.imread(depth_file, -1).astype(np.float32)  # depth is saved 16-bit PNG in millimeters
depth[depth >= 65535] = 0
depth_ = get_depth(depth, calibration_extrinsics, intrinsics_color, intrinsics_depth_inv)

# Plot the depth pixels with value more than zero on the RGB image
for y in range(depth_.shape[0]):
    for x in range(depth_.shape[1]):
        # filter depth<=0
        if depth_[y, x] > 0:
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

# Display the image
cv2.imshow('Depth Map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
