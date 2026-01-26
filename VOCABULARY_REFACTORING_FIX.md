# Vocabulary Refactoring After Balcony Removal - January 22, 2026

## Problem Summary

After removing balconies from the dataset on January 19, 2026 (to fix layout generation issues), training began failing at early epochs (2-5) with NaN errors and CUDA assertions: `t >= 0 && t < n_classes failed`.

### Root Cause
**Vocabulary mismatch** between code and data:
- **Code expected**: 18 object types (indices 0-17, including Balcony at index 9)
- **Data contained**: Only 5 unique indices: `[0, 1, 2, 3, 15]`
- **Model created**: `nn.Embedding(18, 128)` and refinement net with 18 output channels
- **Result**: Model tried to access invalid embedding indices → numerical instability → NaN

## Solution Implemented

### 1. Updated Vocabulary (`model/utils.py`)
**File**: `Network/model/utils.py`, lines 73-120

**Changes**:
- Reduced `room_label` from 18 types to 5 types matching data
- Added `data_idx_to_model_idx` mapping dictionary to handle sparse indices
- Maps data indices `[0, 1, 2, 3, 15]` → model indices `[0, 1, 2, 3, 4]` (contiguous)

```python
# Updated vocabulary after balcony removal
room_label = [
    (0, 'LivingRoom', 1, "PublicArea"),
    (1, 'MasterRoom', 0, "Bedroom"),
    (2, 'Kitchen', 1, "FunctionArea"),
    (3, 'Bathroom', 0, "FunctionArea"),
    (15, 'FrontDoor', 1, "Entrance")  # Index 15 in data → remapped to 4
]

# Create remapping: data index → contiguous model index
for model_idx, (data_idx, label, _, _) in enumerate(room_label):
    vocab['data_idx_to_model_idx'][data_idx] = model_idx
```

**Result**: Model now creates `nn.Embedding(5, 128)` with correct size

---

### 2. Index Remapping in Data Loader (`model/floorplan.py`)

**File**: `Network/model/floorplan.py`

#### Change A: `get_rooms()` method (lines 106-113)
Remaps room type indices when loading from data:

```python
def get_rooms(self, tensor=True):
    rooms = self.data.rType
    
    # Remap sparse data indices [0,1,2,3,15] to contiguous [0,1,2,3,4]
    vocab = get_vocab()
    index_map = vocab['data_idx_to_model_idx']
    rooms = np.array([index_map.get(r, r) for r in rooms])
    
    if tensor: rooms = torch.tensor(rooms).long()
    return rooms
```

#### Change B: `get_layout_image()` method (lines 201-233)
- Remaps room types in layout rendering
- Uses index 5 (= num_classes) for background/boundary pixels
- These pixels are ignored in loss calculation via `ignore_index=5`

```python
def get_layout_image(self,tensor=True):
    vocab = get_vocab()
    num_classes = len(vocab['object_idx_to_name'])
    background_idx = num_classes  # 5 for 5 classes
    
    img = np.full((128,128), background_idx, dtype=np.uint8)
    # ... fill rooms ...
    
    # Remap sparse data indices to contiguous model indices
    index_map = vocab['data_idx_to_model_idx']
    rType = np.array([index_map.get(r, r) for r in rType])
    
    # Background/boundary both use index 5 (will be ignored in loss)
    cv2.polylines(img, ..., background_idx)
```

#### Change C: `vis_fp()` function (lines 420-430)
Updated visualization function to remap indices for rendering.

---

### 3. Loss Function Update (`train.py`)

**File**: `Network/train.py`, lines 170-186

**Changes**:
- Dynamic weight tensor size based on vocabulary
- Added `ignore_index` to handle background pixels (index 5)

```python
def get_losses(args):
    vocab = get_vocab()
    num_classes = len(vocab['object_idx_to_name'])  # Now 5, was 18
    
    weight = torch.ones(num_classes).to(device)
    # No longer need to zero out External/ExteriorWall (removed with balconies)
    
    if args.gene_layout: 
        # Ignore background pixels (index 5) in loss calculation
        loss['gene_ce'] = torch.nn.CrossEntropyLoss(
            weight=weight, 
            ignore_index=num_classes  # Ignores index 5
        )
```

---

## Files Modified

1. **`Network/model/utils.py`**
   - Reduced vocabulary from 18 to 5 room types
   - Added index remapping dictionary

2. **`Network/model/floorplan.py`**
   - `get_rooms()`: Remaps room type indices
   - `get_layout_image()`: Remaps layout indices, uses background_idx=5
   - `vis_fp()`: Remaps indices for visualization

3. **`Network/train.py`**
   - `get_losses()`: Dynamic weight tensor size + ignore_index

---

## Vocabulary Mapping

### Data → Model Index Mapping
| Data Index | Room Type | Model Index |
|------------|-----------|-------------|
| 0 | LivingRoom | 0 |
| 1 | MasterRoom | 1 |
| 2 | Kitchen | 2 |
| 3 | Bathroom | 3 |
| 15 | FrontDoor | 4 |
| 5 (computed) | Background/Boundary | *ignored in loss* |

### Removed Types (Previously Indices 4-14, 16-17)
- DiningRoom (4)
- ChildRoom (5)
- StudyRoom (6)
- SecondRoom (7)
- GuestRoom (8)
- **Balcony (9)** ← Primary removal
- Entrance (10)
- Storage (11)
- Wall-in (12)
- External (13)
- ExteriorWall (14)
- InteriorWall (16)
- InteriorDoor (17)

---

## Testing & Validation

### Test Script Created: `test_remapping.py`
Verifies vocabulary remapping:

```bash
cd Network
python test_remapping.py
```

**Output**:
```
======================================================================
VOCABULARY REMAPPING VERIFICATION
======================================================================

1. Model Vocabulary (Contiguous Indices):
   Vocabulary size: 5
   Model indices: [0, 1, 2, 3, 4]
   Object names: ['LivingRoom', 'MasterRoom', 'Kitchen', 'Bathroom', 'FrontDoor']

2. Index Remapping (Data → Model):
   Data  0 → Model 0 (LivingRoom)
   Data  1 → Model 1 (MasterRoom)
   Data  2 → Model 2 (Kitchen)
   Data  3 → Model 3 (Bathroom)
   Data 15 → Model 4 (FrontDoor)

3. Expected Model Behavior:
   ✅ Embedding layer: nn.Embedding(5, 128)
   ✅ Refinement net output: 5 channels
   ✅ CrossEntropyLoss num_classes: 5
   ✅ All indices will be 0-4 (contiguous)

======================================================================
✅ VOCABULARY SUCCESSFULLY UPDATED AND REMAPPED!
======================================================================
```

### Training Validation
**Command**:
```bash
cd Network
python train.py --epoch 6 --learning_rate 1e-5 --batch_size 20 --workers 0
```

**Result**: ✅ **Training runs successfully!**
- Epoch 1 losses: ~0.3 (stable)
- No NaN errors
- No CUDA assertions
- Gradient norms healthy

---

## Key Design Decisions

### Why Remap to Contiguous Indices?
**Option A**: Keep sparse indices [0,1,2,3,15] → requires `nn.Embedding(16, 128)` (wasteful)
**Option B**: Remap to contiguous [0,1,2,3,4] → `nn.Embedding(5, 128)` (efficient) ✅

We chose **Option B** for:
- Memory efficiency (no unused embeddings)
- Cleaner model architecture
- Easier debugging (contiguous 0-4 range)

### Why ignore_index=5 for Background?
The ground truth layout images contain:
- Room pixels: indices 0-4 (room types)
- Background pixels: index 5 (external/boundary areas)

CrossEntropyLoss with `ignore_index=5` ensures:
- Loss only computed on actual room pixels
- Background pixels don't affect gradients
- Matches original behavior (old indices 13, 14 were also ignored)

---

## Future Considerations

### 1. Vocabulary Management
Consider creating a **single source of truth** for room types:
- Centralized configuration file
- Automated validation during data preprocessing
- Runtime checks to catch mismatches early

### 2. Data Preprocessing
Add vocabulary validation to `DataPreparation/` scripts:
```python
def validate_vocabulary(data_mat, vocab_json):
    """Ensure all data indices exist in vocabulary"""
    data_indices = extract_unique_indices(data_mat)
    vocab_indices = load_vocab_indices(vocab_json)
    assert data_indices.issubset(vocab_indices), "Vocabulary mismatch!"
```

### 3. Model Flexibility
Make vocabulary size a **configuration parameter**:
```python
parser.add_argument('--num_objects', default=None, type=int,
                    help='Number of object types (auto-detected from vocab if None)')
```

### 4. Testing
Add **unit tests** for vocabulary consistency:
- Test remapping logic
- Test layout generation
- Test model initialization with different vocabulary sizes

---

## Lessons Learned

1. **Data-Code Coupling**: Hardcoded vocabularies create fragile dependencies
2. **Validation is Critical**: Dataset changes should trigger automated checks
3. **Sparse vs Dense Indexing**: Remapping improves efficiency and reduces bugs
4. **Error Messages**: CUDA assertions can be cryptic; always check data ranges first

---

## Success Criteria Met

✅ Model initializes with correct vocabulary size (5)
✅ Embedding layer accepts indices 0-4
✅ Layout ground truth uses valid indices (0-5)
✅ CrossEntropyLoss ignores background pixels
✅ Training proceeds past epoch 5 without NaN
✅ Gradient norms remain healthy (< 1.0 after clipping)
✅ No CUDA assertions or runtime errors

**Status**: ✅ **RESOLVED** - Ready for full 200-epoch training!

---

## Next Steps

1. **Run full training**: `--epoch 200` with same hyperparameters
2. **Monitor first 10 epochs**: Ensure no late-emerging NaN issues
3. **Compare to baseline**: Check if 5-class model performs similarly to 18-class
4. **Update documentation**: Record new vocabulary in project README

---

## Contact & Troubleshooting

If similar issues occur:
1. Check data indices: `python check_objects.py`
2. Verify vocabulary: `python test_remapping.py`
3. Enable CUDA debugging: `CUDA_LAUNCH_BLOCKING=1 python train.py ...`
4. Check logs for: "Assertion `t >= 0 && t < n_classes` failed"

**Date**: January 22, 2026  
**Fixed By**: AI Assistant (GitHub Copilot)  
**Affected Files**: 3 (utils.py, floorplan.py, train.py)  
**Lines Changed**: ~60 lines
