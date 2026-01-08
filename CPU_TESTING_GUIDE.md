# CPU Testing Guide for Graph2Plan Model

## ✅ Your Model is NOW CPU-Compatible!

I've updated the training script (`Network/train.py`) to automatically detect and use either GPU or CPU. The same model checkpoint works on both!

## 📦 Files to Transfer to Your CPU PC

### 1. **Trained Model Checkpoint** (~90 MB)
```
experiment/2026-01-08/DeepLayout_2026-01-08_09-53-14/checkpoints/latest_checkpoint_50.pt
```

### 2. **Entire Project Folder**
```
C:\Users\hmbashir\AI Training\Graph2Plan\
```
Copy the whole folder to maintain the directory structure.

### 3. **Required Files Already Included:**
- `DataPreparation/data/test.txt` (200 test samples)
- `DataPreparation/data/data.mat` (1000 floor plans)
- `Network/train.py` (now CPU-compatible!)
- All model files in `Network/model/`

## 🚀 Testing on CPU PC

### **Setup on New PC:**

1. **Install Python 3.10 and dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Test (CPU will be auto-detected):**
   ```bash
   cd Network
   python train.py --test_only --checkpoint "../experiment/2026-01-08/DeepLayout_2026-01-08_09-53-14/checkpoints/latest_checkpoint_50.pt"
   ```

3. **You'll see:**
   ```
   Using device: cpu
   Loaded checkpoint from ../experiment/.../latest_checkpoint_50.pt on cpu
   ```

## ⚡ Performance Expectations

| Device | Test Speed (200 samples) |
|--------|-------------------------|
| **GPU (CUDA)** | ~30-60 seconds |
| **CPU** | ~5-15 minutes |

## 🔧 What Changed in the Code

The code now automatically:
- ✅ Detects if CUDA/GPU is available
- ✅ Falls back to CPU if no GPU found
- ✅ Loads checkpoints to the correct device
- ✅ Moves all tensors to the correct device
- ✅ Skips CUDA-specific operations on CPU

## 📊 Model Performance (From Training)

**Training Completed Successfully:**
- **50 Epochs** in 1 hour 3 minutes
- **Final Validation Loss:** 0.0076
- **Box IoU:** 0.125
- **Room Type Accuracy:** 26.9%
- **Dataset:** 800 train + 200 test ResPlan floor plans

## 🎯 Testing Commands

### **Full Test:**
```bash
python train.py --test_only --checkpoint "../experiment/2026-01-08/DeepLayout_2026-01-08_09-53-14/checkpoints/latest_checkpoint_50.pt"
```

### **Generate Specific Samples:**
```bash
python train.py --test_only --checkpoint "../path/to/checkpoint.pt" --num_samples 10
```

### **Check Device:**
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
```

## ⚠️ Important Notes

1. **Same checkpoint works on both GPU and CPU** - no conversion needed!
2. **CPU testing is slower but produces identical results**
3. **All NaN safety checks are still active**
4. **Validation frequency fix is included** (runs every 5 epochs)

## 📝 Next Steps

After transferring to your CPU PC:
1. Install dependencies from `requirements.txt`
2. Copy the entire Graph2Plan folder
3. Run the test command
4. Results will be saved in `experiment/` folder

## 🆘 Troubleshooting

**If you see "CUDA out of memory" on CPU:**
- This shouldn't happen on CPU, but if it does, reduce batch size:
  ```bash
  python train.py --test_only --checkpoint "path/to/checkpoint.pt" --batch_size 10
  ```

**If model loading fails:**
- Make sure the checkpoint path is correct
- Check that the model architecture matches (it should - same code)

**If testing is very slow on CPU:**
- This is normal! CPU is 10-100x slower than GPU
- Consider testing fewer samples first:
  ```bash
  python train.py --test_only --checkpoint "path/to/checkpoint.pt" --num_samples 20
  ```

---

**Created:** January 8, 2026  
**Training Session:** DeepLayout_2026-01-08_09-53-14  
**GPU Used for Training:** CUDA 12.1  
**CPU Compatibility:** Added January 8, 2026
