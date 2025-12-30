# Training Your ResPlan Model - Quick Start Guide

## ✅ Setup Complete!

All necessary changes have been implemented for 18-type training:

1. **Vocabulary updated** - Now includes room types 0-17 (Interface/model/utils.py and Network/model/utils.py)
2. **Loss weights updated** - Changed from 15 to 18 types (Network/train.py)
3. **Model architecture** - Automatically adapts to 18 types via get_vocab()
4. **Data compatibility** - FloorPlan class handles data format conversion

## 🚀 How to Start Training

### Basic Training Command:
```bash
cd "c:\Users\hmbashir\AI Training\Graph2Plan\Network"
python train.py
```

### Recommended Training Command:
```bash
python train.py --batch_size 20 --epoch 150 --learning_rate 0.0001
```

### With All Options:
```bash
python train.py \
    --batch_size 20 \
    --epoch 150 \
    --learning_rate 0.0001 \
    --gene_layout 1 \
    --box_refine 1 \
    --save_interval 5
```

## 📊 What the Training Does

The training script will:
1. Load your 14,446 floor plans from `data/data_train.mat`
2. Split into train/validation sets automatically
3. Train the model for 150 epochs (or your specified number)
4. Save checkpoints every 5 epochs
5. Save the best model based on validation loss

## 📁 Output Files

Training will create an experiment directory like:
```
../experiment/2025-12-30/DeepLayout_[timestamp]/
├── checkpoints/
│   ├── epoch_model_5.pth          # Checkpoint at epoch 5
│   ├── epoch_model_10.pth         # Checkpoint at epoch 10
│   ├── ...
│   ├── latest_model.pth           # Most recent model
│   └── loss_model.pth             # Best model (lowest loss)
├── logs/
│   ├── log.txt                    # Text log
│   └── events.out.tfevents.*      # TensorBoard logs
└── output/
    └── output_*.pkl                # Test results
```

## 📈 Monitoring Training

### Option 1: Watch the Console
The training will show progress bars with:
- Current epoch
- Training loss
- Validation metrics (IoU, accuracy)

### Option 2: Use TensorBoard (Optional)
```bash
tensorboard --logdir="../experiment/2025-12-30/DeepLayout_[timestamp]/logs"
```

## ⏱️ Expected Training Time

- **Per epoch**: ~5-10 minutes (14,446 samples)
- **Full training (150 epochs)**: ~12-25 hours
- **Convergence**: Usually around epoch 100-120

## 🛑 If You Need to Stop Training

Press `Ctrl+C` to stop. You can resume later with:
```bash
python train.py --pretrain ../experiment/.../checkpoints/latest_model.pth
```

## 💾 Reducing GPU Memory Usage

If you get "CUDA out of memory" errors:
```bash
python train.py --batch_size 16    # Reduce batch size
# or
python train.py --batch_size 12
# or
python train.py --batch_size 8
```

## ✅ After Training Completes

1. **Find your best model**:
   - Located at: `../experiment/.../checkpoints/loss_model.pth`
   - This is the model with lowest validation loss

2. **Copy to Interface** (to use for generation):
   ```bash
   copy ..\experiment\2025-12-30\DeepLayout_[timestamp]\checkpoints\loss_model.pth ..\Interface\model\model.pth
   ```

3. **Test the model**:
   - Start the Django interface
   - Try generating floor plans
   - Layouts should now better utilize the boundary space

## 🐛 Troubleshooting

### "RuntimeError: CUDA out of memory"
→ Reduce batch size: `--batch_size 12` or `--batch_size 8`

### "Loss is NaN"
→ Reduce learning rate: `--learning_rate 0.00001`

### Training is too slow
→ Check if GPU is being used (should see "Using device: cuda" in output)

### Validation loss not improving
→ Train longer (200+ epochs) or check data quality

## 📝 Training Parameters Explained

| Parameter | Default | Recommended | Purpose |
|-----------|---------|-------------|---------|
| `--batch_size` | 20 | 20 | Number of samples per batch |
| `--epoch` | 101 | 150 | Total training epochs |
| `--learning_rate` | 0.0001 | 0.0001 | Optimizer learning rate |
| `--gene_layout` | 1 | 1 | Enable layout generation |
| `--box_refine` | 1 | 1 | Enable box refinement |
| `--save_interval` | 5 | 5 | Save checkpoint every N epochs |

## 🎯 What Success Looks Like

Good training progress:
```
Epoch [10/150] Loss: 2.45 | Val Loss: 2.52 | Gene Acc: 0.68
Epoch [20/150] Loss: 1.89 | Val Loss: 1.95 | Gene Acc: 0.74
Epoch [30/150] Loss: 1.54 | Val Loss: 1.61 | Gene Acc: 0.79
...
Epoch [100/150] Loss: 0.85 | Val Loss: 0.92 | Gene Acc: 0.88
```

## 📊 Your Dataset Summary

- **Room types present**: 0, 1, 2, 3, 9, 15 (6 types)
- **Total samples**: 14,446 floor plans
- **Total rooms**: 139,400 rooms
- **Most common**: Bathroom (30.71%), Bedroom (24.95%)
- **Model vocabulary**: 18 types (includes unused types for compatibility)

## 🚦 Ready to Go!

Everything is configured. Just run:
```bash
python train.py --batch_size 20 --epoch 150
```

The training will start immediately and run for approximately 12-25 hours.
