'''
helper functions to train robust feature extractors
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from math import pi
import cv2
import lpips
from math import exp
from torch.autograd import Variable
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rotation


# from pykalman import KalmanFilter

def get_depth(depth, calibration_extrinsics, intrinsics_color, intrinsics_depth_inv):
    """Return the calibrated depth image (7-Scenes).
    Calibration parameters from DSAC (https://github.com/cvlab-dresden/DSAC) used.
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
    ccoord_depth = np.dot(intrinsics_depth_inv, pcoord_depth) * depth
    ccoord_depth[1, :] = - ccoord_depth[1, :]
    ccoord_depth[2, :] = - ccoord_depth[2, :]
    ccoord_depth = np.concatenate((ccoord_depth, ones), axis=0)
    ccoord_color = np.dot(calibration_extrinsics, ccoord_depth)
    ccoord_color = ccoord_color[0:3, :]
    ccoord_color[1, :] = - ccoord_color[1, :]
    ccoord_color[2, :] = depth

    pcoord_color = np.dot(intrinsics_color, ccoord_color)
    pcoord_color = pcoord_color[:, pcoord_color[2, :] != 0]
    pcoord_color[0, :] = pcoord_color[0, :] / pcoord_color[2, :] + 0.5
    pcoord_color[0, :] = pcoord_color[0, :].astype(int)
    pcoord_color[1, :] = pcoord_color[1, :] / pcoord_color[2, :] + 0.5
    pcoord_color[1, :] = pcoord_color[1, :].astype(int)
    pcoord_color = pcoord_color[:, pcoord_color[0, :] >= 0]
    pcoord_color = pcoord_color[:, pcoord_color[1, :] >= 0]
    pcoord_color = pcoord_color[:, pcoord_color[0, :] < img_width]
    pcoord_color = pcoord_color[:, pcoord_color[1, :] < img_height]

    depth_[pcoord_color[1, :].astype(int),
    pcoord_color[0, :].astype(int)] = pcoord_color[2, :]
    return depth_


def freeze_bn_layer(model):
    ''' freeze bn layer by not require grad but still behave differently when model.train() vs. model.eval() '''
    print("Freezing BatchNorm Layers...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # print("this is a BN layer:", module)
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
    return model


def freeze_bn_layer_train(model):
    ''' set batchnorm to eval() 
        it is useful to align train and testing result 
    '''
    # model.train()
    # print("Freezing BatchNorm Layers...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return model


def save_image_saliancy(tensor, path, normalize: bool = False, scale_each: bool = False, ):
    """
    Modification based on TORCHVISION.UTILS
    ::param: tensor (batch, channel, H, W)
    """
    # grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=32)
    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=6)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    fig = plt.figure()
    plt.imshow(ndarr[:, :, 0], cmap='jet')  # viridis, plasma
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.close()


def save_image_saliancy_single(tensor, path, normalize: bool = False, scale_each: bool = False, ):
    """
    Modification based on TORCHVISION.UTILS, save single feature map
    ::param: tensor (batch, channel, H, W)
    """
    # grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=32)
    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=1)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    fig = plt.figure()
    # plt.imshow(ndarr[:,:,0], cmap='plasma') # viridis, jet
    plt.imshow(ndarr[:, :, 0], cmap='jet')  # viridis, jet
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.close()


def print_feature_examples(features, path):
    """
    print feature maps
    ::param: features
    """
    kwargs = {'normalize': True, }  # 'scale_each' : True

    for i in range(len(features)):
        fn = path + '{}.png'.format(i)
        # save_image(features[i].permute(1,0,2,3), fn, **kwargs)
        save_image_saliancy(features[i].permute(1, 0, 2, 3), fn, normalize=True)
    # pdb.set_trace()
    ###


def plot_features(features, path='f', isList=True):
    """
    print feature maps
    :param features: (3, [batch, H, W]) or [3, batch, H, W]
    :param path: save image path
    :param isList: wether the features is an list
    :return:
    """
    kwargs = {'normalize': True, }  # 'scale_each' : True

    if isList:
        dim = features[0].dim()
    else:
        dim = features.dim()
    assert (dim == 3 or dim == 4)

    if dim == 4 and isList:
        print_feature_examples(features, path)
    elif dim == 4 and (isList == False):
        fn = path
        lvl, b, H, W = features.shape
        for i in range(features.shape[0]):
            fn = path + '{}.png'.format(i)
            save_image_saliancy(features[i][None, ...].permute(1, 0, 2, 3).cpu(), fn, normalize=True)

        # # concat everything
        # features = features.reshape([-1, H, W])
        # # save_image(features[None,...].permute(1,0,2,3).cpu(), fn, **kwargs)
        # save_image_saliancy(features[None,...].permute(1,0,2,3).cpu(), fn, normalize=True) 

    elif dim == 3 and isList:  # print all images in the list
        for i in range(len(features)):
            fn = path + '{}.png'.format(i)
            # save_image(features[i][None,...].permute(1,0,2,3).cpu(), fn, **kwargs)
            save_image_saliancy(features[i][None, ...].permute(1, 0, 2, 3).cpu(), fn, normalize=True)
    elif dim == 3 and (isList == False):
        fn = path
        save_image_saliancy(features[None, ...].permute(1, 0, 2, 3).cpu(), fn, normalize=True)


def sample_homography_np(
        shape, shift=0, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi / 2,
        allow_artifacts=False, translation_overflow=0.):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography. (like crop size)
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # print("debugging")

    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                              [patch_ratio, patch_ratio], [patch_ratio, 0]])

    from numpy.random import uniform
    from scipy.stats import truncnorm

    # Random perspective and affine perturbations
    # lower, upper = 0, 2
    std_trunc = 2
    # pdb.set_trace()
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y / 2).rvs(1)
        h_displacement_left = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x / 2).rvs(1)
        h_displacement_right = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x / 2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1 * std_trunc, std_trunc, loc=1, scale=scaling_amplitude / 2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx, :, :]

    # Random translation
    if translation:
        # pdb.set_trace()
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0], 1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul((pts2 - center)[np.newaxis, :, :], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else:  # find multiple valid option and choose the valid one
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx, :, :]

    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 *= shape[np.newaxis, :]
    pts2 *= shape[np.newaxis, :]

    homography = cv2.getPerspectiveTransform(np.float32(pts1 + shift), np.float32(pts2 + shift))
    return homography


def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies

    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.view(batch_size * 3, 3)

    warped_points = homographies @ points.transpose(0, 1)

    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0, :, :] if no_batches else warped_points


def inv_warp_image_batch(img, mat_homo_inv, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    '''
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1, 1, img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1, 3, 3)

    Batch, channel, H, W = img.shape
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing='ij'), dim=2)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([Batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()

    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img


def compute_valid_mask(image_shape, inv_homography, device='cpu', erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        `erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """

    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)
    batch_size = inv_homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0], image_shape[1]).to(device)
    mask = inv_warp_image_batch(mask, inv_homography, device=device, mode='nearest')
    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    mask = mask.cpu().numpy()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius * 2,) * 2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)


def get_interest_region_cords(cords_list, H, W):
    interest_regions = np.zeros((H, W), dtype=np.uint8)
    interest_regions[cords_list[:, 1], cords_list[:, 0]] = 1
    interest_regions = cv2.dilate(interest_regions, np.ones((10, 10), np.uint8),
                                  iterations=1)  # adjust the size of the interest region
    interest_regions = np.array(interest_regions, dtype=bool)
    coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1), dtype=int)
    interest_indices = coords[interest_regions]

    return interest_indices  # (N, 2)


def find_POI(img_rgb, DEBUG=False):  # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    if DEBUG:
        img = cv2.drawKeypoints(img_gray, keypoints, img)
        cv2.imwrite("./sift.png", img)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy  # pixel coordinates


lpips_alex = lpips.LPIPS(net='alex').to('cuda')  # best forward scores
lpips_vgg = lpips.LPIPS(net='vgg').to('cuda')  # closer to "traditional" perceptual loss, when used for optimization


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def lpips(img1, img2, net='alex', format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    # Ensure images are on the same device as the LPIPS model
    img1 = img1.to('cuda')
    img2 = img2.to('cuda')

    if net == 'alex':
        return lpips_alex(img1, img2)
    elif net == 'vgg':
        return lpips_vgg(img1, img2)


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))