"""
Verify that model now uses correct number of channels (18 instead of 15)
"""
import sys
sys.path.append('.')
import torch
from model.model import Model
from model.utils import get_vocab

print("="*70)
print("VERIFYING MODEL ARCHITECTURE FIX")
print("="*70)

# Check vocabulary size
vocab = get_vocab()
num_objs = len(vocab['object_idx_to_name'])
print(f"\nVocabulary size: {num_objs} room types")

# Create model with default settings
print("\nCreating model with default settings (box_refine_arch=None)...")
model = Model()

# Check if box_refine_backbone exists
if model.box_refine_backbone is not None:
    print("✅ box_refine_backbone exists")
    
    # Get the first Conv2d layer
    first_layer = model.box_refine_backbone[0]
    if isinstance(first_layer, torch.nn.Conv2d):
        in_channels = first_layer.in_channels
        out_channels = first_layer.out_channels
        kernel_size = first_layer.kernel_size
        
        print(f"\nFirst Conv2d layer:")
        print(f"  Input channels: {in_channels}")
        print(f"  Output channels: {out_channels}")
        print(f"  Kernel size: {kernel_size}")
        
        if in_channels == num_objs:
            print(f"\n✅ SUCCESS: Input channels ({in_channels}) matches vocab size ({num_objs})")
        else:
            print(f"\n❌ FAIL: Input channels ({in_channels}) does not match vocab size ({num_objs})")
    else:
        print(f"⚠️  First layer is not Conv2d: {type(first_layer)}")
else:
    print("❌ box_refine_backbone is None")

# Test with a dummy forward pass
print("\n" + "-"*70)
print("Testing forward pass with 18-channel layout...")
print("-"*70)

try:
    # Create dummy input matching what the model generates
    batch_size = 4
    height = 128
    width = 128
    
    # Simulate generated layout with 18 channels
    dummy_layout = torch.randn(batch_size, num_objs, height, width)
    print(f"\nInput shape: {dummy_layout.shape}")
    print(f"  Batch: {batch_size}")
    print(f"  Channels: {num_objs}")
    print(f"  Height: {height}")
    print(f"  Width: {width}")
    
    # Pass through box_refine_backbone
    if model.box_refine_backbone is not None:
        output = model.box_refine_backbone(dummy_layout)
        print(f"\nOutput shape: {output.shape}")
        print(f"✅ Forward pass successful!")
    else:
        print("⚠️  Cannot test - box_refine_backbone is None")
        
except RuntimeError as e:
    if "expected input" in str(e) and "channels" in str(e):
        print(f"\n❌ CHANNEL MISMATCH ERROR:")
        print(f"   {e}")
    else:
        print(f"\n❌ ERROR: {e}")
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

if model.box_refine_backbone is not None:
    first_layer = model.box_refine_backbone[0]
    if isinstance(first_layer, torch.nn.Conv2d) and first_layer.in_channels == num_objs:
        print("\n🎉 Model is correctly configured for 18-type vocabulary!")
        print("   You can now resume training.")
    else:
        print("\n⚠️  Model may have issues. Check the output above.")
else:
    print("\n⚠️  box_refine_backbone not initialized.")

print("="*70)
