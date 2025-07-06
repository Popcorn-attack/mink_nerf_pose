import os
import sys
import torch
import torch.nn as nn

# Add the script directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from models.minkloc_multimodal import MinkLocMultimodal, ResnetFeatureExtractor
from models.minkloc import MinkLoc

def test_minkloc_multimodal():
    # Test parameters
    batch_size = 2
    cloud_fe_size = 256
    image_fe_size = 512
    output_dim = 256
    
    # Create mock feature extractors
    class MockCloudFeatureExtractor(nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.output_size = output_size
            
        def forward(self, batch):
            return {'embedding': torch.randn(batch_size, self.output_size)}
    
    # Initialize feature extractors
    cloud_fe = MockCloudFeatureExtractor(cloud_fe_size)
    image_fe = ResnetFeatureExtractor(image_fe_size, add_fc_block=True)
    
    # Test different fusion methods
    fusion_methods = ['concat', 'add']
    
    for fuse_method in fusion_methods:
        print(f"\nTesting MinkLocMultimodal with {fuse_method} fusion method")
        
        # For 'add' fusion, we need equal feature sizes
        if fuse_method == 'add':
            # Create new feature extractors with equal sizes
            cloud_fe = MockCloudFeatureExtractor(image_fe_size)
            cloud_fe_size = image_fe_size
        
        # Initialize model
        model = MinkLocMultimodal(
            cloud_fe=cloud_fe,
            cloud_fe_size=cloud_fe_size,
            image_fe=image_fe,
            image_fe_size=image_fe_size,
            output_dim=output_dim,
            fuse_method=fuse_method,
            dropout_p=0.3,
            final_block='mlp'
        )
        
        # Create mock input batch
        batch = {
            'images': torch.randn(batch_size, 3, 480, 640),  # RGB images
            'cloud': torch.randn(batch_size, 1000, 3)  # Point cloud data
        }
        
        # Forward pass
        output = model(batch)
        
        # Verify output structure
        assert 'embedding' in output, "Output should contain 'embedding' key"
        assert 'image_embedding' in output, "Output should contain 'image_embedding' key"
        assert 'cloud_embedding' in output, "Output should contain 'cloud_embedding' key"
        
        # Verify output shapes
        assert output['embedding'].shape == (batch_size, output_dim), \
            f"Final embedding shape should be ({batch_size}, {output_dim})"
        assert output['image_embedding'].shape == (batch_size, image_fe_size), \
            f"Image embedding shape should be ({batch_size}, {image_fe_size})"
        assert output['cloud_embedding'].shape == (batch_size, cloud_fe_size), \
            f"Cloud embedding shape should be ({batch_size}, {cloud_fe_size})"
        
        # Test model info printing
        model.print_info()
        
        print(f"Successfully tested MinkLocMultimodal with {fuse_method} fusion method")

def test_minkloc_multimodal_edge_cases():
    # Test parameters
    batch_size = 2
    cloud_fe_size = 256
    image_fe_size = 256  # Same size for add fusion
    output_dim = 256
    
    # Create mock feature extractors
    class MockCloudFeatureExtractor(nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.output_size = output_size
            
        def forward(self, batch):
            return {'embedding': torch.randn(batch_size, self.output_size)}
    
    # Test cases
    test_cases = [
        {
            'name': 'Cloud only',
            'cloud_fe': MockCloudFeatureExtractor(cloud_fe_size),
            'image_fe': None,
            'fuse_method': 'concat'
        },
        {
            'name': 'Image only',
            'cloud_fe': None,
            'image_fe': ResnetFeatureExtractor(image_fe_size, add_fc_block=True),
            'fuse_method': 'concat'
        },
        {
            'name': 'Add fusion with equal sizes',
            'cloud_fe': MockCloudFeatureExtractor(image_fe_size),
            'image_fe': ResnetFeatureExtractor(image_fe_size, add_fc_block=True),
            'fuse_method': 'add'
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting edge case: {test_case['name']}")
        
        try:
            model = MinkLocMultimodal(
                cloud_fe=test_case['cloud_fe'],
                cloud_fe_size=cloud_fe_size if test_case['cloud_fe'] else 0,
                image_fe=test_case['image_fe'],
                image_fe_size=image_fe_size if test_case['image_fe'] else 0,
                output_dim=output_dim,
                fuse_method=test_case['fuse_method'],
                dropout_p=0.3,
                final_block='mlp'
            )
            
            # Create mock input batch
            batch = {
                'images': torch.randn(batch_size, 3, 480, 640) if test_case['image_fe'] else None,
                'cloud': torch.randn(batch_size, 1000, 3) if test_case['cloud_fe'] else None
            }
            
            # Forward pass
            output = model(batch)
            
            # Verify output structure
            assert 'embedding' in output, "Output should contain 'embedding' key"
            if test_case['image_fe']:
                assert 'image_embedding' in output, "Output should contain 'image_embedding' key"
            if test_case['cloud_fe']:
                assert 'cloud_embedding' in output, "Output should contain 'cloud_embedding' key"
            
            # Verify output shape
            assert output['embedding'].shape == (batch_size, output_dim), \
                f"Final embedding shape should be ({batch_size}, {output_dim})"
            
            print(f"Successfully tested edge case: {test_case['name']}")
            
        except Exception as e:
            print(f"Failed to test edge case {test_case['name']}: {str(e)}")
            raise

if __name__ == '__main__':
    print("Testing MinkLocMultimodal model...")
    test_minkloc_multimodal()
    test_minkloc_multimodal_edge_cases()
    print("\nAll tests completed successfully!") 