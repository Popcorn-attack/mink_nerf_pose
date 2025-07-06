import os
import argparse
import numpy as np
import cv2
import struct
import open3d as o3d
import json
import glob

def downsample_pointcloud(points, target_points=4096):
    """
    Downsample point cloud to a specific number of points using voxel grid downsampling
    followed by random sampling if needed.
    
    Args:
        points: (N, 3) numpy array of points
        target_points: target number of points (default: 4096)
    
    Returns:
        downsampled_points: (target_points, 3) numpy array
    """
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate voxel size based on point cloud bounds
    bounds = pcd.get_axis_aligned_bounding_box()
    diagonal = np.linalg.norm(bounds.get_extent())
    voxel_size = diagonal / np.cbrt(target_points * 2)  # Multiply by 2 to ensure we get enough points
    
    # Perform voxel grid downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    
    # Convert back to numpy array
    downsampled_points = np.asarray(downsampled_pcd.points)
    
    # If we have more points than target, randomly sample
    if len(downsampled_points) > target_points:
        indices = np.random.choice(len(downsampled_points), target_points, replace=False)
        downsampled_points = downsampled_points[indices]
    # If we have fewer points than target, randomly duplicate points
    elif len(downsampled_points) < target_points:
        indices = np.random.choice(len(downsampled_points), target_points - len(downsampled_points), replace=True)
        downsampled_points = np.vstack([downsampled_points, downsampled_points[indices]])
    
    return downsampled_points

def depth_to_pointcloud(depth_img, intrinsic, depth_scale=1000.0):
    """
    Convert depth image to point cloud
    :param depth_img: depth image (16-bit single channel)
    :param intrinsic: camera intrinsic parameters [fx, fy, cx, cy]
    :param depth_scale: depth scale factor (default 1000, for Kinect-style data)
    :return: (N, 3) point cloud array
    """
    fx, fy, cx, cy = intrinsic
    height, width = depth_img.shape
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    
    # Convert to real depth (in meters)
    z = depth_img.astype(np.float32) / depth_scale
    
    # Calculate 3D coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Flatten and combine point cloud
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
    # Remove invalid points (depth=0)
    valid_mask = (z > 0).reshape(-1)
    points = points[valid_mask]
    
    # Downsample to 4096 points
    points = downsample_pointcloud(points)
    
    return points

def save_ply(points, filename, binary=True):
    """
    Save point cloud to PLY file
    :param points: (N, 3) point cloud array
    :param filename: output filename
    :param binary: whether to save in binary format
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd, write_ascii=not binary)

def process_depth_image(depth_path, output_path, intrinsic, depth_scale=1000.0):
    """
    Process a single depth image
    :param depth_path: path to depth image
    :param output_path: path to save PLY file
    :param intrinsic: camera intrinsics
    :param depth_scale: depth scale factor
    """
    # Read depth image
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if depth_img is None:
        print(f"Warning: Could not read {depth_path}, skipping")
        return
    
    # Convert to point cloud
    points = depth_to_pointcloud(depth_img, intrinsic, depth_scale)
    
    # Save point cloud
    save_ply(points, output_path)
    print(f"Saved: {output_path} ({len(points)} points)")

def process_7scenes_dataset(dataset_dir, scene_name):
    """
    Process a 7Scenes dataset scene
    :param dataset_dir: root directory of 7Scenes dataset
    :param scene_name: name of the scene (e.g., 'heads', 'chess', etc.)
    """
    # 7Scenes camera intrinsics
    intrinsic = [585.0, 585.0, 320.0, 240.0]  # fx, fy, cx, cy
    
    # Scene directory
    scene_dir = os.path.join(dataset_dir, scene_name)
    
    # Process each sequence
    sequences = sorted(glob.glob(os.path.join(scene_dir, 'seq-*')))
    for seq_dir in sequences:
        # Skip zip files
        if '.zip' in seq_dir:
            continue
            
        print(f"\nProcessing sequence: {os.path.basename(seq_dir)}")
        
        # Find all depth images in the sequence
        depth_files = glob.glob(os.path.join(seq_dir, '*depth.png'))
        
        for depth_file in depth_files:
            # Generate output path (replace .depth.png with .ply)
            output_file = depth_file.replace('.depth.png', '.ply')
            
            # Process the depth image
            process_depth_image(
                depth_path=depth_file,
                output_path=output_file,
                intrinsic=intrinsic,
                depth_scale=1000.0
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate PLY point clouds from 7Scenes depth images')
    parser.add_argument('--dataset_dir', type=str, default='/home/qiqi/APR/data/7scenes',
                        help='Root directory of 7Scenes dataset')
    parser.add_argument('--scene_name', type=str, default='heads',
                        help='Name of the scene (e.g., heads, chess, etc.)')
    
    args = parser.parse_args()
    
    # Process the specified scene
    process_7scenes_dataset(
        dataset_dir=args.dataset_dir,
        scene_name=args.scene_name
    )