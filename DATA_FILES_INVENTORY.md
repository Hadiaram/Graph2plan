# Data Files Inventory - Git LFS Tracking

## Summary
- **Total Size**: ~876 MB
- **Total Files**: 13 large files (>1 MB each)
- **Git LFS Tracking**: ALL data files (.mat, .pkl, .npy)

---

## 1. Network Data Files (242 MB)
**Purpose**: Training, validation, and test datasets for Graph2Plan model

| File | Size | Status | Description |
|------|------|--------|-------------|
| `Network/data/data_train.mat` | 51.26 MB | ✅ Staged | Training dataset |
| `Network/data/data_valid.mat` | 19.13 MB | ✅ Staged | Validation dataset |
| `Network/data/data_test.mat` | 56.45 MB | ✅ Staged | Test dataset |
| `Network/data.mat` | 57.88 MB | ⏳ To stage | Additional training data |

**LFS Pattern**: `*.mat`, `**/*.mat`

---

## 2. DataPreparation Files (527 MB)
**Purpose**: Preprocessing, cleaning, and data transformation artifacts

### Main Files
| File | Size | Status | Description |
|------|------|--------|-------------|
| `DataPreparation/cleaned_resplan.pkl` | 292.28 MB | ⏳ To stage | **LARGEST** - Cleaned ResPlan dataset |
| `DataPreparation/data.pkl` | 89.50 MB | ⏳ To stage | Intermediate processed data |

### Data Subdirectory
| File | Size | Status | Description |
|------|------|--------|-------------|
| `DataPreparation/data/D_test_train.npy` | 140.52 MB | ⏳ To stage | Distance matrices |
| `DataPreparation/data/trainTF.pkl` | 4.46 MB | ⏳ To stage | Training features |

**LFS Patterns**: `*.pkl`, `**/*.pkl`, `*.npy`, `**/*.npy`

---

## 3. Interface Files (223 MB)
**Purpose**: Web interface data - retrieval, clustering, and converted datasets

### Retrieval Files
| File | Size | Status | Description |
|------|------|--------|-------------|
| `Interface/retrieval/tf_train.npy` | 110.21 MB | ⏳ To stage | Training features for retrieval |
| `Interface/retrieval/clusters_train.npy` | 7.63 MB | ⏳ To stage | Cluster assignments |
| `Interface/retrieval/centroids_train.npy` | 3.81 MB | ⏳ To stage | Cluster centroids |

### Static Data Files
| File | Size | Status | Description |
|------|------|--------|-------------|
| `Interface/static/Data/data_test_converted.pkl` | 56.29 MB | ⏳ To stage | Converted test data |
| `Interface/static/Data/data_train_converted.pkl` | 44.53 MB | ⏳ To stage | Converted training data |

**LFS Patterns**: `*.npy`, `**/*.npy`, `**/*.pkl`

---

## Git LFS Commands

### Quick Setup (All Files)
```bash
# Run the automated script
setup_all_lfs.bat
```

### Manual Setup
```bash
# Initialize Git LFS
git lfs install

# Track all data file types
git lfs track "*.mat"
git lfs track "**/*.mat"
git lfs track "*.pkl"
git lfs track "**/*.pkl"
git lfs track "*.npy"
git lfs track "**/*.npy"

# Stage .gitattributes
git add .gitattributes

# Stage all data files
git add Network/data/*.mat Network/data.mat
git add DataPreparation/*.pkl DataPreparation/data/*.npy DataPreparation/data/*.pkl
git add Interface/retrieval/*.npy Interface/static/Data/*.pkl

# Verify
git lfs ls-files

# Commit
git commit -m "Add all data files with Git LFS tracking"

# Push (this uploads ~876 MB to LFS)
git push origin ai-training-branch
```

---

## GitHub LFS Limits (Free Tier)
- **Storage**: 1 GB total
- **Bandwidth**: 1 GB/month
- **Current Usage**: ~876 MB (87% of limit)

### ⚠️ Important Notes
1. **First push will use ~876 MB bandwidth** - this is your entire monthly allowance
2. **Remaining storage after push**: ~148 MB (15%)
3. **Future updates**: Each modified file counts against monthly bandwidth
4. **Recommendation**: Only push when necessary, avoid frequent updates to large files

---

## File Categories

### 🔴 Critical Files (Cannot Regenerate)
- `Network/data/*.mat` - Core datasets
- `Interface/static/Data/*.pkl` - Converted data for web interface

### 🟡 Important Files (Can Regenerate with Effort)
- `DataPreparation/cleaned_resplan.pkl` - Takes time to regenerate
- `Interface/retrieval/*.npy` - Clustering results

### 🟢 Regeneratable Files (Can Rebuild)
- `DataPreparation/data/*.npy` - Distance matrices (can recalculate)
- `DataPreparation/data.pkl` - Intermediate processing

---

## Alternative Strategy (If LFS Limit Is Concern)

If you want to stay under 500 MB, track only critical files:

```bash
# Track only essential files
git lfs track "Network/data/*.mat"
git lfs track "Network/data.mat"
git lfs track "Interface/static/Data/*.pkl"

# Ignore regeneratable files in .gitignore
echo "DataPreparation/cleaned_resplan.pkl" >> .gitignore
echo "DataPreparation/data.pkl" >> .gitignore
echo "DataPreparation/data/*.npy" >> .gitignore
echo "Interface/retrieval/*.npy" >> .gitignore
```

**Essential files only**: ~286 MB (29% of limit)

---

## Verification Commands

```bash
# Check what's tracked by LFS
git lfs ls-files

# Check file sizes before committing
git lfs ls-files --size

# See LFS status
git lfs status

# Check your GitHub LFS quota
git lfs env
```

---

## Created: 2025-01-05
## Last Updated: 2025-01-05
