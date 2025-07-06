import set_sys_path
import os, sys
import json
import math
import imageio
import random
import time
import wandb

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
from train_mink_depth_nerf import load_pretrained_minkloc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_on_epoch(model, train_dl, optimizer, loss_fn, device, epoch, writer, args):
    """
    Train for one epoch
    """
    model.train()
    # Set batch norm layers to eval mode
    def set_bn_eval(m):
        if isinstance(m,torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, ME.MinkowskiBatchNorm):
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
        optimizer.zero_grad()
        output = model(batch)
        pred_pose_ = output['embedding']
        # Calculate loss and backpropagate
        loss = loss_fn(output['embedding'], pose)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        pred_pose_ = pred_pose_.detach().cpu() 
        batch_pose = batch['pose'].detach().cpu() 
        batch_size = pred_pose_.shape[0]
        #write as the validate() function to print the tansition and rotation error
        errors = get_error_in_q(pred_pose_.reshape(batch_size, 3, 4), batch_pose.reshape(batch_size, 3, 4), batch_size=batch_size)
        #calculate the median of errors
        median_errors = torch.median(errors, dim=0)[0]
        #calculate the mean of errors
        mean_errors = torch.mean(errors, dim=0)
        print(f"mean_errors: {mean_errors}")
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{train_loss/(batch_idx+1):.4f}',
            'pos_error': f'{median_errors[0]:.2f}m',
            'rot_error': f'{median_errors[1]:.2f}°' 
        })
        
        # Log training progress
        if batch_idx % args.i_print == 0:
            writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_dl) + batch_idx)
            writer.add_scalar('Loss/train_avg', train_loss/(batch_idx+1), epoch * len(train_dl) + batch_idx)
            
            # Log detailed metrics to wandb every i_print steps
            wandb.log({
                "batch_loss": loss.item(),
                "batch_avg_loss": train_loss/(batch_idx+1),
                "batch_position_error": median_errors[0].item(),
                "batch_rotation_error": median_errors[1].item(),
                "batch_position_error_mean": mean_errors[0].item(),
                "batch_rotation_error_mean": mean_errors[1].item(),
                "step": epoch * len(train_dl) + batch_idx
            })
    
    return train_loss / len(train_dl)

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion
    Args:
        R: rotation matrix (3x3)
    Returns:
        q: quaternion (4,)
    """
    # Convert to numpy for easier calculation
    R = R.cpu().numpy()
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

def validate(model, val_dl, loss_fn, device):
    """
    Validate the model
    """
    model.eval()
    val_loss = 0.0
    all_errors = []
    
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
            loss = loss_fn(output['embedding'], pose)
            val_loss += loss.item()
            
            # Reshape predictions and targets to 3x4 matrices
            pred_pose = output['embedding'].reshape(-1, 3, 4)
            gt_pose = pose.reshape(-1, 3, 4)
            
            # Calculate errors
            errors = get_error_in_q(pred_pose, gt_pose, batch_size=pred_pose.shape[0])
            all_errors.append(errors)
            
            # Update progress bar
            current_errors = torch.cat(all_errors, dim=0)
            median_errors = torch.median(current_errors, dim=0)[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{val_loss/(pbar.n+1):.4f}',
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
    
    return val_loss / len(val_dl), median_errors

def train_posenet(args, train_dl, val_dl, model, max_epochs, optimizer, loss_fn, scheduler, device, early_stopping):
    """
    Train the MinkLocMultimodal model
    """
    # Initialize wandb
    wandb.init(
        project="direct-posenet",
        name=f"{args.model_name}_{args.dataset_type}",
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": max_epochs,
            "model_name": args.model_name,
            "dataset": args.dataset_type,
            "scene": args.scene if hasattr(args, 'scene') else 'unknown',
            "optimizer": "Adam",
            "loss_function": "MSE",
        }
    )
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.basedir, args.model_name))
    
    # Initialize best model tracking
    best_median_error = float('inf')
    args.model_name = 'minkloc_pose'
    best_model_path = os.path.join(args.basedir, args.model_name, 'best_model.pth')
    
    # Training loop
    for epoch in range(max_epochs):
        # Train for one epoch
        avg_train_loss = train_on_epoch(model, train_dl, optimizer, loss_fn, device, epoch, writer, args)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        
        # Log training metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Validate every 2 epochs
        if epoch % 2 == 0:
            avg_val_loss, median_errors = validate(model, val_dl, loss_fn, device)
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

def evaluate(model, test_dl, loss_fn, device, args):
    """
    Evaluate the model on test set
    """
    # Load the best model
    best_model_path = os.path.join(args.basedir, args.model_name, 'best_model.pth')
    
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}")
        return
    
    print(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Best model was saved at epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['loss']:.4f}")
    print(f"Best position error: {checkpoint['position_error']:.2f}m")
    print(f"Best rotation error: {checkpoint['rotation_error']:.2f}°")
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    all_errors = []
    all_predictions = []
    all_ground_truths = []
    
    print("\nEvaluating on test set...")
    pbar = tqdm(test_dl, desc='Testing', leave=True)
    
    with torch.no_grad():
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
            loss = loss_fn(output['embedding'], pose)
            test_loss += loss.item()
            
            # Reshape predictions and targets to 3x4 matrices
            pred_pose = output['embedding'].reshape(-1, 3, 4)
            gt_pose = pose.reshape(-1, 3, 4)
            
            # Store predictions and ground truths for detailed analysis
            all_predictions.append(pred_pose.cpu())
            all_ground_truths.append(gt_pose.cpu())
            
            # Calculate errors
            errors = get_error_in_q(pred_pose, gt_pose, batch_size=pred_pose.shape[0])
            all_errors.append(errors)
            
            # Update progress bar
            current_errors = torch.cat(all_errors, dim=0)
            median_errors = torch.median(current_errors, dim=0)[0]
            mean_errors = torch.mean(current_errors, dim=0)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{test_loss/(batch_idx+1):.4f}',
                'pos_error': f'{median_errors[0]:.2f}m',
                'rot_error': f'{median_errors[1]:.2f}°'
            })
    
    # Calculate final statistics
    all_errors = torch.cat(all_errors, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truths = torch.cat(all_ground_truths, dim=0)
    
    median_errors = torch.median(all_errors, dim=0)[0]
    mean_errors = torch.mean(all_errors, dim=0)
    std_errors = torch.std(all_errors, dim=0)
    
    # Calculate percentiles
    pos_errors = all_errors[:, 0]
    rot_errors = all_errors[:, 1]
    
    quantile_values = torch.tensor([0.25, 0.5, 0.75, 0.9, 0.95, 0.99], device=pos_errors.device)
    pos_percentiles = torch.quantile(pos_errors, quantile_values)
    rot_percentiles = torch.quantile(rot_errors, quantile_values)
    
    # Print comprehensive results
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total test samples: {len(all_errors)}")
    print(f"Average test loss: {test_loss/len(test_dl):.4f}")
    print(f"\nPosition Error (meters):")
    print(f"  Mean: {mean_errors[0]:.3f} ± {std_errors[0]:.3f}")
    print(f"  Median: {median_errors[0]:.3f}")
    print(f"  25th percentile: {pos_percentiles[0]:.3f}")
    print(f"  75th percentile: {pos_percentiles[2]:.3f}")
    print(f"  90th percentile: {pos_percentiles[3]:.3f}")
    print(f"  95th percentile: {pos_percentiles[4]:.3f}")
    print(f"  99th percentile: {pos_percentiles[5]:.3f}")
    
    print(f"\nRotation Error (degrees):")
    print(f"  Mean: {mean_errors[1]:.3f} ± {std_errors[1]:.3f}")
    print(f"  Median: {median_errors[1]:.3f}")
    print(f"  25th percentile: {rot_percentiles[0]:.3f}")
    print(f"  75th percentile: {rot_percentiles[2]:.3f}")
    print(f"  90th percentile: {rot_percentiles[3]:.3f}")
    print(f"  95th percentile: {rot_percentiles[4]:.3f}")
    print(f"  99th percentile: {rot_percentiles[5]:.3f}")
    print(f"{'='*60}")
    
    # Save detailed results
    results = {
        'test_loss': test_loss/len(test_dl),
        'position_error': {
            'mean': mean_errors[0].item(),
            'median': median_errors[0].item(),
            'std': std_errors[0].item(),
            'percentiles': pos_percentiles.tolist()
        },
        'rotation_error': {
            'mean': mean_errors[1].item(),
            'median': median_errors[1].item(),
            'std': std_errors[1].item(),
            'percentiles': rot_percentiles.tolist()
        },
        'all_errors': all_errors.cpu().numpy(),
        'all_predictions': all_predictions.cpu().numpy(),
        'all_ground_truths': all_ground_truths.cpu().numpy()
    }
    
    results_path = os.path.join(args.basedir, args.model_name, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, torch.Tensor) else x)
    
    print(f"\nDetailed results saved to: {results_path}")
    
    # Log test results to wandb
    if wandb.run is not None:
        wandb.log({
            "test_loss": results['test_loss'],
            "test_position_error_mean": results['position_error']['mean'],
            "test_position_error_median": results['position_error']['median'],
            "test_position_error_std": results['position_error']['std'],
            "test_rotation_error_mean": results['rotation_error']['mean'],
            "test_rotation_error_median": results['rotation_error']['median'],
            "test_rotation_error_std": results['rotation_error']['std'],
            "test_position_95th_percentile": results['position_error']['percentiles'][4],
            "test_rotation_95th_percentile": results['rotation_error']['percentiles'][4]
        })
    
    return results

def main():
    """
    Main function to run training
    """
    # Parse arguments
    parser = config_parser()
    args = parser.parse_args()
    
    print(parser.format_values())
    
    # Load data using the NeRF dataloader
    train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader_NeRF(args)
    
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
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
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
    if args.continue_training:  
        model, optimizer, start_epoch = load_pretrained_minkloc(args, model, optimizer)
    
    ####Train the model
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
        early_stopping=early_stopping
    )

    # Evaluate the model on test set
    evaluate(model, val_dl, loss_fn, device, args)

if __name__ == '__main__':
    main() 