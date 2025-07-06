import os
import json
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import cv2
import imageio
from script.utils.utils import get_depth, find_POI, get_interest_region_cords
import transforms3d.quaternions as txq
# see for formulas:
# https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-801-machine-vision-fall-2004/readings/quaternions.pdf
# and "Quaternion and Rotation" - Yan-Bin Jia, September 18, 2016
from dataset_loaders.utils.color import rgb_to_yuv
from plyfile import PlyData

def RT2QT(poses_in, mean_t, std_t):
    """
    processes the 1x12 raw pose from dataset by aligning and then normalizing
    :param poses_in: N x 12
    :param mean_t: 3
    :param std_t: 3
    :return: processed poses (translation + quaternion) N x 7
    """
    poses_out = np.zeros((len(poses_in), 7))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(R)
        q = q / (np.linalg.norm(q) + 1e-12)  # normalize
        q *= np.sign(q[0])  # constrain to hemisphere
        poses_out[i, 3:] = q

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out


def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


def process_poses_rotmat(poses_in, mean_t, std_t, align_R, align_t, align_s):
    """
    processes the 1x12 raw pose from dataset by aligning and then normalizing
    produce logq
    :param poses_in: N x 12
    :return: processed poses N x 12
    """
    return poses_in


def process_poses_q(poses_in, mean_t, std_t, align_R, align_t, align_s):
    """
    processes the 1x12 raw pose from dataset by aligning and then normalizing
    produce logq
    :param poses_in: N x 12
    :param mean_t: 3
    :param std_t: 3
    :param align_R: 3 x 3
    :param align_t: 3
    :param align_s: 1
    :return: processed poses (translation + log quaternion) N x 6
    """
    poses_out = np.zeros((len(poses_in), 6))  # (1000,6)
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]  # x,y,z position
    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]  # rotation
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere, first number, +1/-1, q.shape (1,4)
        poses_out[i, 3:] = q  # logq rotation
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t  # (1000, 6)
    poses_out[:, :3] /= std_t
    return poses_out


def process_poses_logq(poses_in, mean_t, std_t, align_R, align_t, align_s):
    """
    processes the 1x12 raw pose from dataset by aligning and then normalizing
    produce logq
    :param poses_in: N x 12
    :param mean_t: 3
    :param std_t: 3
    :param align_R: 3 x 3
    :param align_t: 3
    :param align_s: 1
    :return: processed poses (translation + log quaternion) N x 6
    """
    poses_out = np.zeros((len(poses_in), 6))  # (1000,6)
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]  # x,y,z position
    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]  # rotation
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere, first number, +1/-1, q.shape (1,4)
        q = qlog(q)  # (1,3)
        poses_out[i, 3:] = q  # logq rotation
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t  # (1000, 6)
    poses_out[:, :3] /= std_t
    return poses_out


from torchvision.datasets.folder import default_loader


def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    return img


def load_depth_image(filename):
    try:
        img_depth = Image.fromarray(np.array(Image.open(filename)).astype("uint16"))
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    return img_depth


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def normalize_recenter_pose(poses, sc, hwf):
    ''' normalize xyz into [-1, 1], and recenter pose '''
    target_pose = poses.reshape(poses.shape[0], 3, 4)
    target_pose[:, :3, 3] = target_pose[:, :3, 3] * sc

    x_norm = target_pose[:, 0, 3]
    y_norm = target_pose[:, 1, 3]
    z_norm = target_pose[:, 2, 3]

    tpose_ = target_pose + 0

    # find the center of pose
    center = np.array([x_norm.mean(), y_norm.mean(), z_norm.mean()])
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])

    # pose avg
    vec2 = normalize(tpose_[:, :3, 2].sum(0))
    up = tpose_[:, :3, 1].sum(0)
    hwf = np.array(hwf).transpose()
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)

    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [tpose_.shape[0], 1, 1])
    poses = np.concatenate([tpose_[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    return poses[:, :3, :].reshape(poses.shape[0], 12)


class SevenScenes(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None,
                 target_transform=None, mode=0, seed=7,
                 df=1., trainskip=1, testskip=1, hwf=[480, 640, 585.],
                 ret_idx=False, fix_idx=False, ret_hist=False, hist_bin=10):
        """
        :param scene: scene name ['chess', 'pumpkin', ...]
        :param data_path: root 7scenes data directory.
        Usually '../data/deepslam_data/7Scenes'
        :param train: if True, return the training images. If False, returns the
        testing images
        :param transform: transform to apply to the images
        :param target_transform: transform to apply to the poses
        :param mode: (Obsolete) 0: just color image, 1: color image in NeRF 0-1 and resized.
        :param df: downscale factor
        :param trainskip: due to 7scenes are so big, now can use less training sets # of trainset = 1/trainskip
        :param testskip: skip part of testset, # of testset = 1/testskip
        :param hwf: H,W,Focal from COLMAP
        :param ret_idx: bool, currently only used by NeRF-W
        """
        self.n_points = 4096
        self.transform = transform
        self.target_transform = target_transform
        self.df = df

        self.H, self.W, self.focal = hwf
        self.H = int(self.H)
        self.W = int(self.W)
        np.random.seed(seed)

        self.train = train
        self.ret_idx = ret_idx
        self.fix_idx = fix_idx
        self.ret_hist = ret_hist
        self.hist_bin = hist_bin  # histogram bin size

        # configs for calibrated depth (https://github.com/AaltoVision/hscnet/blob/master/datasets/seven_scenes.py)
        self.intrinsics_color = np.array([[525.0, 0.0, 320.0],
                                          [0.0, 525.0, 240.0],
                                          [0.0, 0.0, 1.0]])

        self.intrinsics_depth = np.array([[585.0, 0.0, 320.0],
                                          [0.0, 585.0, 240.0],
                                          [0.0, 0.0, 1.0]])

        self.intrinsics_depth_inv = np.linalg.inv(self.intrinsics_depth)
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        self.calibration_extrinsics = np.loadtxt(
            os.path.join('/home/qiqi/APR/NeRF-Pose/data', '7Scenes', 'sensorTrans.txt'))

        # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)
        data_dir = osp.join('/home/qiqi/APR/NeRF-Pose/data', '7Scenes', scene)
        world_setup_fn = data_dir + '/world_setup.json'
        # read json file
        with open(world_setup_fn, 'r') as myfile:
            data = myfile.read()

        # parse json file
        obj = json.loads(data)
        self.near = obj['near']
        self.far = obj['far']
        self.pose_scale = obj['pose_scale']
        self.pose_scale2 = obj['pose_scale2']
        self.move_all_cam_vec = obj['move_all_cam_vec']

        # decide which sequences to use
        if train:
            split_file = osp.join(base_dir, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]  # parsing

        # read poses and collect image names
        self.c_imgs = []
        self.d_imgs = []
        self.pcds = []
        self.interest_indices_list = []  # for key-point of image
        self.gt_idx = np.empty((0,), dtype=np.int32)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))

            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
            idxes = [int(n[6:12]) for n in p_filenames]

            frame_idx = np.array(sorted(idxes))

            # trainskip and testskip
            if train and trainskip > 1:
                frame_idx_tmp = frame_idx[::trainskip]
                frame_idx = frame_idx_tmp
            elif not train and testskip > 1:
                frame_idx_tmp = frame_idx[::testskip]
                frame_idx = frame_idx_tmp

            pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))).flatten()[:12] for i in
                   frame_idx]  # 3x4 pose matrices
            ps[seq] = np.asarray(pss)
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.gt_idx = np.hstack((self.gt_idx, gt_offset + frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i)) for i in frame_idx]
            d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i)) for i in frame_idx]
            pcds = [osp.join(seq_dir, 'frame-{:06d}.depth.ply'.format(i)) for i in frame_idx]

            self.c_imgs.extend(c_imgs)
            self.d_imgs.extend(d_imgs)
            self.pcds.extend(pcds)

        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train:
            mean_t = np.zeros(3)
            std_t = np.ones(3)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert pose to translation + log quaternion
        logq = False
        quat = False
        if logq:  # (batch_num, 6)
            self.poses = np.empty((0, 6))
        elif quat:  # (batch_num, 7)
            self.poses = np.empty((0, 7))
        else:  # (batch_num, 12)
            self.poses = np.empty((0, 12))

        for seq in seqs:
            if logq:
                pss = process_poses_logq(poses_in=ps[seq], mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'],
                                         align_t=vo_stats[seq]['t'],
                                         align_s=vo_stats[seq]['s'])  # here returns t + logQed R
                self.poses = np.vstack((self.poses, pss))
            elif quat:
                pss = RT2QT(poses_in=ps[seq], mean_t=mean_t, std_t=std_t)  # here returns t + quaternion R
                self.poses = np.vstack((self.poses, pss))
            else:
                pss = process_poses_rotmat(poses_in=ps[seq], mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'],
                                           align_t=vo_stats[seq]['t'],
                                           align_s=vo_stats[seq]['s'])
                self.poses = np.vstack((self.poses, pss))

        # debug read one img and get the shape of the img
        img = load_image(self.c_imgs[0])
        img_np = (np.array(img) / 255.).astype(np.float32)  # (480,640,3)
        self.H, self.W = img_np.shape[:2]
        if self.df != 1.:
            self.H = int(self.H // self.df)
            self.W = int(self.W // self.df)
            self.focal = self.focal / self.df

        # # get key-point for images by SIFT
        # for p in self.c_imgs:
        #     img = imageio.imread(p)[:, :, :3]
        #     img_resize = cv2.resize(img, (self.W, self.H))  # (120, 160, 3)
        #     cords_list = find_POI(img_resize)
        #     interest_indices = get_interest_region_cords(cords_list, self.H, self.W)
        #     self.interest_indices_list.append(interest_indices)

        print('')

    def __len__(self):
        return self.poses.shape[0]
    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_path = filename
        pc = PlyData.read(file_path)
        pc = np.array(pc.elements[0].data)
        
        # Get the x, y, z coordinates from the point cloud data
        x = pc['x']
        y = pc['y']
        z = pc['z']
        
        # Stack coordinates into Nx3 matrix
        pc = np.stack([x, y, z], axis=1)
        
        # Ensure we have exactly 4096 points
        if len(pc) > self.n_points:
            # Randomly sample points if we have too many
            indices = np.random.choice(len(pc), self.n_points, replace=False)
            pc = pc[indices]
        elif len(pc) < self.n_points:
            # Randomly duplicate points if we have too few
            indices = np.random.choice(len(pc), self.n_points - len(pc), replace=True)
            pc = np.vstack([pc, pc[indices]])
            
        pc = torch.tensor(pc, dtype=torch.float)
        return pc
    def __getitem__(self, index):
        results = {}

        # key-point for image
        # interest_indices = self.interest_indices_list[index]

        # get RGB image
        img = load_image(self.c_imgs[index])  # original img.size(w,h) = (640,480)

        # get calibrated depth
        depth = cv2.imread(self.d_imgs[index], -1).astype(np.float32)
        depth[depth == 65535] = 0.0  # filter invalid depth==65535
        depth_calibrate = get_depth(depth, self.calibration_extrinsics, self.intrinsics_color,
                                    self.intrinsics_depth_inv)

        pose = self.poses[index]
        pcd = self.load_pc(self.pcds[index])

        if self.df != 1.:  # resize image and depth
            img_np = (np.array(img) / 255.0).astype(np.float32)
            depth_np = (depth_calibrate / 1000.0).astype(np.float32) # depth in meters
            dims = (self.W, self.H)
            img_half_res = cv2.resize(img_np, dims, interpolation=cv2.INTER_AREA)  # (H, W, 3)
            depth_half_res = cv2.resize(depth_np, dims, interpolation=cv2.INTER_AREA)  # (H, W, 1)
            img = img_half_res
            depth = depth_half_res

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.transform is not None:
            img = self.transform(img)

        if self.ret_idx:
            if self.train and self.fix_idx == False:
                return img, pose, index
            else:
                return img, pose, 0

        # results['interest_indices'] = interest_indices # Note,just used in "run_nerf.py", while delete it in "run_feature.py"
        results['img_path'] = os.path.join(
            os.path.basename(os.path.dirname(self.c_imgs[index])),  # Extract "seq-03"
            os.path.basename(self.c_imgs[index])  # Extract "frame-000000.color.png"
        )
        results['img'] = img
        results['depth'] = depth
        results['pose'] = pose
        results['pcd'] = pcd
        

        if self.ret_hist:
            yuv = rgb_to_yuv(img)
            y_img = yuv[0]  # extract y channel only
            hist = torch.histc(y_img, bins=self.hist_bin, min=0., max=1.)  # compute intensity histogram, each value in hist represents the number of pixels falling into that bin
            hist = hist / (hist.sum()) * 100  # convert to histogram density, in terms of percentage per bin
            hist = torch.round(hist)
            results['hist'] = hist
            return results
        else:
            return results
