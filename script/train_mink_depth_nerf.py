import set_sys_path
import os, sys
import json
import math
import imageio
import random
import time
import wandb
from copy import deepcopy

# Add the root directory to the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from callbacks import EarlyStopping
import MinkowskiEngine as ME
import transforms3d

from models.minkloc_multimodal import MinkLocMultimodal, ResnetFeatureExtractor
from models.minkloc import MinkLoc
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader_NeRF
from options_new import config_parser
from models.rendering import *
from models.nerfw import *
from losses import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_on_epoch(model, train_dl, optimizer, loss_fn, device, epoch, writer, args, hwf, near, far, i_split, N_rand, render_kwargs_test):
    """
    Train for one epoch
    """
    model.train()
    def set_bn_eval(m):
        if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, ME.MinkowskiBatchNorm):
            m.eval()
            m.track_running_stats = False
    model.apply(set_bn_eval)
    train_loss = 0.0
    # Create progress bar
    pbar = tqdm(train_dl, desc=f'Epoch {epoch}', leave=True)
    
    for batch_idx, data in enumerate(pbar):
        # Process input data
        img = data['img'].to(device)
        pcd = data['pcd'].to(device)
        hist = data['hist'].to(device)
        pose = data['pose'].to(device)
        depth = data['depth'].to(device)
        # Process point cloud
        points = pcd.cpu().numpy()
        
        # Quantize coordinates and create batched coordinates
        coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=0.01) for e in points]
        coords = ME.utils.batched_coordinates(coords) # On CPU
        
        # Keep features on CPU for MinkowskiEngine
        features = torch.ones((coords.shape[0], 1), dtype=torch.float32) # On CPU
        
        # Create batch dictionary
        batch = {
            'images': img,
            'coords': coords,  # On CPU
            'features': features,  # On CPU
            'hist': hist,
            'depth': depth,
            'pose': pose
        }
        
        # Forward pass
        output = model(batch)
        
        img = deepcopy(batch['images'])
        batch_size = img.shape[0]
        pred_pose_ = output['embedding'] 
        pred_pose = pred_pose_.clone()
        # pose_gt = batch['pose'] 
        ###print the comparison of pred_pose and pose_gt values one line by line
        # print(f"Comparison of pred_pose and pose_gt values:")
        # for i in range(batch_size):
        #     print(f"pred_pose: {pred_pose[i]}")
        #     print(f"pose_gt: {batch['pose'][i]}")
        #     print("--------------------------------")
        
        # Unpack hwf for render function
        H, W, focal = hwf
        iter_psnr = 0
        num_psnr = 0
        # loss_func = NerfWLoss(coef=1, lambda_u=0.01)
        render_loss = torch.tensor(0.0, device=device)
        use_pred = True
        for i in range(batch_size):
            img_target = img[i].permute(1, 2, 0).to(device)
            if args.dataset_type == '7Scenes':
                depth_target = batch['depth'][i].to(device)

            if use_pred:
                pose = pred_pose[i].reshape(3, 4).to(device)  # reshape to 3x4 rot matrix
            else:
                pose = batch['pose'][i].reshape(3, 4).to(device)
            # Fix histogram indexing
            img_idx = hist[i].to(device)  # Keep as tensor and add batch dimension
            # pose = pose.unsqueeze(0)
            # pose = fix_coord_supp(args, pose, world_setup_dict, device=device)
            pose_pred = torch.zeros(1, 4, 4).to(device)
            pose_pred[0, :3, :4] = pose[:3, :4]  # (1,3,4))
            pose_pred[0, 3, 3] = 1.
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # from code: https://github1s.com/yitongx/sinerf/blob/main/tasks/nerfmm/train_eval_sinerf.py#L131
            rgb, acc, depth, depth_var, extras = render(H//2, W//2, focal/2, chunk=args.chunk, c2w=pose_pred[0,:3, :4], img_idx=img_idx, **render_kwargs_test)
            ####save rgb 
            if use_pred:
                rgbs = rgb.cpu().detach().numpy()
                depths = depth.cpu().detach().numpy()
                rgb_path = os.path.join(args.basedir, args.expname,  f'pred_rendered_rgb{i}.png')
                #if dir not exist, create it
                if not os.path.exists(os.path.dirname(rgb_path)):
                    os.makedirs(os.path.dirname(rgb_path))
                imageio.imwrite(rgb_path, to8b(rgbs))
                depth_path = os.path.join(args.basedir, args.expname, f'pred_rendered_depth{i}.png')

                imageio.imwrite(depth_path, to8b(depths))
            else:
                rgbs = rgb.cpu().detach().numpy()
                depths = depth.cpu().detach().numpy()
                rgb_path = os.path.join(args.basedir, args.expname, f'gt_rendered_rgb{i}.png')
                imageio.imwrite(rgb_path, to8b(rgbs))
                depth_path = os.path.join(args.basedir, args.expname, f'gt_rendered_depth{i}.png')
                imageio.imwrite(depth_path, to8b(depths))
            #calculate mse loss by rgb  
            #resize img_target to rgb.shape,  note that rgb.shape is half of img_target.shape
            img_target = img_target[::2, ::2, :]
            depth_target = depth_target[::2, ::2]

            rgb_loss = img2mse(rgb, img_target)
            depth_loss = img2mse(depth,  depth_target)
            batch_loss = rgb_loss + depth_loss    
            
            render_loss = render_loss + batch_loss
            
            with torch.no_grad():
                img_loss = img2mse(rgb, img_target)
                psnr = mse2psnr(img_loss)
                iter_psnr += psnr.item()
                num_psnr += 1   
        
        psnr = iter_psnr / num_psnr
        # psnr = 0    
        render_loss = render_loss / batch_size
        
        # Reset default tensor type
        torch.set_default_tensor_type('torch.FloatTensor')
        
        pose_loss = loss_fn(pred_pose_, batch['pose'])
        loss =   0.7*pose_loss +0.3*render_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        
        #detach pred_pose_ and batch['pose']
        pred_pose_ = pred_pose_.detach().cpu() 
        batch_pose = batch['pose'].detach().cpu() 
        #write as the validate() function to print the tansition and rotation error
        errors = get_error_in_q(pred_pose_.reshape(batch_size, 3, 4), batch_pose.reshape(batch_size, 3, 4), batch_size=batch_size)
        # print(f"errors: {errors}")
        #calculate the median of errors
        median_errors = torch.median(errors, dim=0)[0]
        # print(f"median_errors: {median_errors}")
        #calculate the mean of errors
        mean_errors = torch.mean(errors, dim=0)
        # print(f"mean_errors: {mean_errors}")
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{train_loss/(batch_idx+1):.4f}',
            'psnr': f'{psnr:.2f}',
            'pos_error': f'{median_errors[0]:.2f}m',
            'rot_error': f'{median_errors[1]:.2f}°'
        })
        
        # Log training progress
        if batch_idx % args.i_print == 0:
            writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_dl) + batch_idx)
            writer.add_scalar('Loss/train_avg', train_loss/(batch_idx+1), epoch * len(train_dl) + batch_idx)
            writer.add_scalar('Loss/pose_loss', pose_loss.item(), epoch * len(train_dl) + batch_idx)
            writer.add_scalar('Loss/render_loss', render_loss.item(), epoch * len(train_dl) + batch_idx)
            writer.add_scalar('PSNR/train_step', psnr, epoch * len(train_dl) + batch_idx)
            
            # Log detailed metrics to wandb every i_print steps
            wandb.log({
                "batch_loss": loss.item(),
                "batch_avg_loss": train_loss/(batch_idx+1),
                "batch_pose_loss": pose_loss.item(),
                "batch_render_loss": render_loss.item(),
                "batch_psnr": psnr,
                "batch_position_error": median_errors[0].item(),
                "batch_rotation_error": median_errors[1].item(),
                "batch_position_error_mean": mean_errors[0].item(),
                "batch_rotation_error_mean": mean_errors[1].item(),
                "step": epoch * len(train_dl) + batch_idx
            })
    
    return train_loss / len(train_dl), psnr

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion
    Args:
        R: rotation matrix (3x3)
    Returns:
        q: quaternion (4,)
    """
    # Convert to numpy for easier calculation
    R = R.detach().cpu().numpy()
    return torch.tensor(transforms3d.quaternions.mat2quat(R))

def get_error_in_q(pred_pose, gt_pose, batch_size=1):
    """
    Calculate position and rotation errors
    Args:
        pred_pose: predicted pose (batch_size, 3, 4)
        gt_pose: ground truth pose (batch_size, 3, 4)
        batch_size: batch size
    Returns:
        errors: tensor of position and rotation errors (batch_size, 2)
    """
    device = pred_pose.device
    errors = torch.zeros((batch_size, 2), device=device)
    
    # Extract rotation matrices and translations
    gt_R = gt_pose[:, :3, :3]
    gt_t = gt_pose[:, :3, 3]
    pred_R = pred_pose[:, :3, :3]
    pred_t = pred_pose[:, :3, 3]
    
    # Convert rotation matrices to quaternions
    gt_q = torch.stack([rotation_matrix_to_quaternion(gt_R[i]) for i in range(batch_size)])
    pred_q = torch.stack([rotation_matrix_to_quaternion(pred_R[i]) for i in range(batch_size)])
    
    # Calculate errors for each sample
    for i in range(batch_size):
        # Normalize quaternions
        q1 = gt_q[i] / torch.norm(gt_q[i])
        q2 = pred_q[i] / torch.norm(pred_q[i])
        
        # Calculate rotation error
        d = torch.abs(torch.dot(q1, q2))
        d = torch.clamp(d, -1.0, 1.0)  # acos can only input [-1~1]
        theta = 2 * torch.acos(d) * 180 / math.pi
        
        # Calculate position error
        error_x = torch.norm(gt_t[i] - pred_t[i])
        
        errors[i] = torch.tensor([error_x, theta], device=device)
    
    return errors

def validate(model, val_dl, loss_fn, device, args, hwf, near, far, i_split, N_rand, render_kwargs_train):
    """
    Validate the model
    """
    model.eval()
    val_loss = 0.0
    all_errors = []
    iter_psnr = 0
    num_psnr = 0
    
    # Create progress bar for validation
    pbar = tqdm(val_dl, desc='Validation', leave=True)
    
    with torch.no_grad():
        for data in pbar:
            # Process input data
            img = data['img'].to(device)
            pcd = data['pcd'].to(device)
            hist = data['hist'].to(device)
            pose = data['pose'].to(device)
            depth = data['depth'].to(device)
            
            # Process point cloud
            points = pcd.cpu().numpy()
            
            # Quantize coordinates and create batched coordinates
            coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=0.01) for e in points]
            coords = ME.utils.batched_coordinates(coords)  # On CPU
            
            # Keep features on CPU for MinkowskiEngine
            features = torch.ones((coords.shape[0], 1), dtype=torch.float32)  # On CPU
            
            # Create batch dictionary
            batch = {
                'images': img,
                'coords': coords,  # On CPU
                'features': features,  # On CPU
                'hist': hist,
                'depth': depth,
                'pose': pose
            }
            
            output = model(batch)
            
            img = deepcopy(batch['images'])
            batch_size = img.shape[0]
            pred_pose_ = output['embedding']
            pred_pose = pred_pose_.clone()
            
            # Add debug information for comparison with train_on_epoch
            # print(f"VALIDATE - Comparison of pred_pose and pose_gt values:")
            # for i in range(batch_size):
            #     print(f"pred_pose: {pred_pose[i]}")
            #     print(f"pose_gt: {batch['pose'][i]}")
            #     print("--------------------------------")
            
            # Unpack hwf for render function
            H, W, focal = hwf
            render_loss = torch.tensor(0.0, device=device)
            
            for i in range(batch_size):
                img_target = img[i].permute(1, 2, 0).to(device)
                if args.dataset_type == '7Scenes':
                    depth_target = batch['depth'][i].to(device)

                pose = pred_pose[i].reshape(3, 4).to(device)  # reshape to 3x4 rot matrix
                img_idx = hist[i].to(device)  # Keep as tensor and add batch dimension
                
                pose_pred = torch.zeros(1, 4, 4).to(device)
                pose_pred[0, :3, :4] = pose[:3, :4]  # (1,3,4))
                pose_pred[0, 3, 3] = 1.
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                
                # Render using the same approach as training
                rgb, acc, render_depth, depth_var, extras = render(H//2, W//2, focal/2, chunk=args.chunk, 
                                                                c2w=pose_pred[0,:3, :4], img_idx=img_idx, 
                                                                **render_kwargs_train)
                
                #save rgb and depth for each epoch
                
                rgbs = rgb.cpu().detach().numpy()
                depths = render_depth.cpu().detach().numpy()
                rgb_path = os.path.join(args.basedir, args.expname, f'val_pred_rendered_rgb{i}.png')
                imageio.imwrite(rgb_path, to8b(rgbs))
                depth_path = os.path.join(args.basedir, args.expname, f'val_pred_rendered_depth{i}.png')
                imageio.imwrite(depth_path, to8b(depths))
                
                #resize img_target to rgb.shape, note that rgb.shape is half of img_target.shape
                img_target = img_target[::2, ::2, :]
                depth_target = depth_target[::2, ::2]
                
                # Calculate losses
                rgb_loss = img2mse(rgb, img_target)
                depth_loss = img2mse(render_depth, depth_target)
                loss = rgb_loss + depth_loss
                
                render_loss = render_loss + loss
                
                with torch.no_grad():
                    img_loss = img2mse(rgb, img_target)
                    psnr = mse2psnr(img_loss)
                    iter_psnr += psnr.item()
                    num_psnr += 1
            
            psnr = iter_psnr / num_psnr
            # psnr = 0
            render_loss = render_loss / batch_size
            
            # Reset default tensor type
            torch.set_default_tensor_type('torch.FloatTensor')
            
            # gt_pose = batch['pose'].reshape(-1, 3, 4)
            
            loss = render_loss + loss_fn(pred_pose_,  batch['pose'])
            val_loss += loss.item()
            
            # Calculate errors
            errors = get_error_in_q(pred_pose_.reshape(batch_size, 3, 4), batch['pose'].reshape(batch_size, 3, 4), batch_size=batch_size)
            all_errors.append(errors)
            # print(f"errors: {errors}")
            #calculate the median of errors
            median_errors = torch.median(errors, dim=0)[0]
            # print(f"median_errors: {median_errors}")
            #calculate the mean of errors
            mean_errors = torch.mean(errors, dim=0)
            # print(f"mean_errors: {mean_errors}")    
            # Update progress bar
            current_errors = torch.cat(all_errors, dim=0)
            median_errors = torch.median(current_errors, dim=0)[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{val_loss/(pbar.n+1):.4f}',
                'psnr': f'{psnr:.2f}',
                'pos_error': f'{median_errors[0]:.2f}m',
                'rot_error': f'{median_errors[1]:.2f}°'
            })
    
    # Calculate final statistics
    all_errors = torch.cat(all_errors, dim=0)
    median_errors = torch.median(all_errors, dim=0)[0]
    mean_errors = torch.mean(all_errors, dim=0)
    
    print(f'\nValidation Results:')
    print(f'Median error: {median_errors[0]:.2f}m and {median_errors[1]:.2f}°')
    print(f'Mean error: {mean_errors[0]:.2f}m and {mean_errors[1]:.2f}°')
    print(f'Average PSNR: {psnr:.2f}')
    
    return val_loss / len(val_dl), median_errors

def train_posenet(args, train_dl, val_dl, model, max_epochs, optimizer, loss_fn, scheduler, device, early_stopping, hwf, near, far, i_split, N_rand, render_kwargs_test):
    """
    Train the MinkLocMultimodal model with NeRF rendering
    """
    # Initialize wandb
    wandb.init(
        project="minkloc-nerf-pose",
        name=f"{args.model_name}_{args.dataset_type}_{args.scene if hasattr(args, 'scene') else 'unknown'}",
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": max_epochs,
            "near": near,
            "far": far,
            "N_rand": N_rand,
            "chunk": args.chunk,
            "hwf": hwf
        }
    )
    #freeze bn
    model.apply(freeze_bn_layer_train)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.basedir, args.model_name))
    
    # Initialize best model tracking
    best_median_error = float('inf')
    best_model_path = os.path.join(args.basedir, args.model_name, 'best_model.pth')
    
    # Training loop
    for epoch in range(max_epochs):
        # Train for one epoch
        avg_train_loss, psnr = train_on_epoch(model, train_dl, optimizer, loss_fn, device, epoch, writer, args, hwf, near, far, i_split, N_rand, render_kwargs_test)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('PSNR/train_epoch', psnr, epoch)
        
        # Log training metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_psnr": psnr,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            avg_val_loss, median_errors = validate(model, val_dl, loss_fn, device, args, hwf, near, far, i_split, N_rand, render_kwargs_test)
            writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
            writer.add_scalar('Error/position_median', median_errors[0], epoch)
            writer.add_scalar('Error/rotation_median', median_errors[1], epoch)
            
            # Log validation metrics to wandb
            wandb.log({
                "val_loss": avg_val_loss,
                "position_error_median": median_errors[0].item(),
                "rotation_error_median": median_errors[1].item(),
                "position_error_median_m": median_errors[0].item(),
                "rotation_error_median_deg": median_errors[1].item()
            })
            
            # Save best model based on rotation error
            if median_errors[1] < best_median_error:
                best_median_error = median_errors[1]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                    'position_error': median_errors[0].item(),
                    'rotation_error': median_errors[1].item(),
                }, best_model_path)
                print(f'\nNew best model saved! Rotation error: {median_errors[1]:.2f}°')
                
                # Log best model metrics to wandb
                wandb.log({
                    "best_rotation_error": median_errors[1].item(),
                    "best_position_error": median_errors[0].item(),
                    "best_model_epoch": epoch
                })
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Early stopping check
        early_stopping(avg_val_loss, model, epoch=epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            wandb.log({"early_stopping_epoch": epoch})
            break
    
    writer.close()
    wandb.finish()
    print(f'\nTraining completed. Best rotation error: {best_median_error:.2f}°')

def load_pretrained_minkloc(args, model, optimizer):
    """
    Load pretrained MinkLoc model
    Args:
        args: arguments containing model paths and configurations
        model: MinkLoc model to load weights into
        optimizer: optimizer to load state into
    Returns:
        model: model with loaded weights
        optimizer: optimizer with loaded state
        start_epoch: starting epoch
    """
    if args.pretrain_model_path == '':
        print('No pretrained MinkLoc model specified')
        return model, optimizer, 0
    
    print(f'Loading pretrained MinkLoc model from {args.pretrain_model_path}')
    ckpt = torch.load(args.pretrain_model_path)
    
    # Print checkpoint keys for debugging
    print("Checkpoint keys:", ckpt.keys())
    
    # Try different possible keys for model state dict
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        print("Warning: Could not find model state dict in checkpoint. Available keys:", ckpt.keys())
        return model, optimizer, 0
    
    # Get starting epoch
    start_epoch = ckpt.get('epoch', 0)
    
    # If optimizer state exists and we want to continue training
    if 'optimizer_state_dict' in ckpt and args.continue_training:
        print('Loading optimizer state')
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    return model, optimizer, start_epoch
 
def freeze_bn_layer_train(model):
    ''' set batchnorm to eval() 
        it is useful to align train and testing result 
    '''
    # model.train()
    # print("Freezing BatchNorm Layers...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, ME.MinkowskiBatchNorm):
            module.eval()
            module.track_running_stats = False
              
    return model
def main():
    """
    Main function to run training
    """
    # Parse arguments
    parser = config_parser()
    args = parser.parse_args()
    
    print(parser.format_values())
    
    # Load data using the NeRF dataloader
    train_dl, val_dl, hwf, i_split, bds, render_poses, render_img = load_7Scenes_dataloader_NeRF(args)
    near, far = bds
    # Create feature extractors
    cloud_fe = MinkLoc(
        in_channels=1,
        feature_size=256,
        output_dim=256,
        planes=[32, 64, 64],
        layers=[1, 1, 1],
        num_top_down=1,
        conv0_kernel_size=5,
        block='BasicBlock',
        pooling_method='GeM',
        linear_block=False,
        dropout_p=0.3
    )
    # Keep cloud feature extractor on CPU
    cloud_fe = cloud_fe.cpu()
    # Ensure all parameters are on CPU
    for param in cloud_fe.parameters():
        param.data = param.data.cpu()
    
    image_fe = ResnetFeatureExtractor(
        output_dim=512,
        add_fc_block=True
    ).to(device)  # Move image feature extractor to GPU
    
    # Create model with feature extractors
    model = MinkLocMultimodal(
        cloud_fe=cloud_fe,
        cloud_fe_size=256,
        image_fe=image_fe,
        image_fe_size=512,
        output_dim=12,  # Changed to 12 for pose output (3 translation + 9 rotation)
        fuse_method='concat',
        dropout_p=0.3,
        final_block='mlp'
    )
    
    # Add final projection layer for pose output (12 dimensions: 3 for translation + 9 for rotation matrix)
    # Input dimension is 768 (256 + 512 from concatenated features)
    model.final_net = nn.Sequential(
        nn.Linear(768, 12),  # First reduce to 256 dimensions  # Output 12 dimensions for pose
    )
    
    # Move model to device, but keep cloud_fe on CPU
    model.to(device)
    # Ensure cloud_fe stays on CPU
    model.cloud_fe = model.cloud_fe.cpu()
    for param in model.cloud_fe.parameters():
        param.data = param.data.cpu()
    
    # Print model info
    model.print_info()
    
    # Set up loss function
    loss_fn = nn.MSELoss()
    
    # Set up optimizer
    
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, shape={tuple(param.shape)}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
     # Load pretrained MinkLoc model if specified
    model, _, start_epoch = load_pretrained_minkloc(args, model, optimizer)
    
    #load nerf model
    # render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    render_kwargs_train, render_kwargs_test, start, grad_vars, _ = create_nerf(args)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.75,
        patience=args.patience[1],
        verbose=True
    )
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        args,
        patience=args.patience[0],
        verbose=True
    )


    # Evaluate NeRF module performance
    print("\nEvaluating NeRF module performance...")
    # evaluate_nerf(args, val_dl, hwf, near, far, i_split, args.N_rand)
    
    # Train the model
    train_posenet(
        args=args,
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        max_epochs=args.max_epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        early_stopping=early_stopping,
        hwf=hwf,
        near=near,
        far=far,
        i_split=i_split,
        N_rand=args.N_rand,
        render_kwargs_test=render_kwargs_test
    )
world_setup_dict = {
    "near":0,
    "far":2.5,
    "pose_scale": 1,
    "pose_scale2": 1,
    "move_all_cam_vec": [0.0, 0.0, 1.0]
    }
if __name__ == '__main__':
    main() 