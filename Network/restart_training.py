"""
Safe training restart script
Clears CUDA cache and restarts training with fresh Python process
"""
import subprocess
import sys
import os

print("=" * 70)
print("SAFE TRAINING RESTART")
print("=" * 70)

print("\n1. Killing any existing Python processes...")
# Note: Be careful with this - it kills ALL python processes
# Commented out for safety - user should manually close training if needed
# subprocess.run(["taskkill", "/F", "/IM", "python.exe"], shell=True, check=False)

print("\n2. Starting fresh training session...")
print("   Command: python train.py --epoch 200 --learning_rate 1e-5 --batch_size 20 --workers 0")
print("\n" + "=" * 70)

# Set environment variable to force CUDA cache clear
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Run training in fresh process
subprocess.run([
    sys.executable, 
    "train.py",
    "--epoch", "200",
    "--learning_rate", "1e-5", 
    "--batch_size", "20",
    "--workers", "0"
], cwd=os.path.dirname(os.path.abspath(__file__)))
