# Train.py - Complete Training Logic Explanation

## Overview
This file implements the **training pipeline** for the Graph2Plan model. It handles data loading, optimization, loss computation, validation, and all epoch-dependent training strategies.

**Key Insight**: This file contains ALL epoch-aware logic. The model itself (model.py) is epoch-agnostic.

**File Size**: 873 lines
**Primary Framework**: PyTorch Ignite (training orchestration)

---

## File Structure

```
Lines 1-35:    Imports
Lines 36-103:  Argument parsing (configuration)
Lines 105-197: Setup functions (model, data, optimizer, losses)
Lines 199-208: Utility functions
Lines 209-724: Main training loop and orchestration
Lines 726-868: Testing phase
Lines 869-872: Entry point
```

---

# Part 1: Configuration (Lines 36-103)

## Lines 36-103: parse_args() - Command Line Arguments

All hyperparameters and settings are configured via command line arguments.

### Dataset Parameters (Lines 39-45)
```python
parser.add_argument('--dataset_dir', default='./data', type=str)
parser.add_argument('--image_size', default='128,128', type=int_tuple)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--with_house', default='0', type=bool_flag)
parser.add_argument('--pos_dim', default=25, type=int)
parser.add_argument('--area_dim', default=10, type=int)
```
- **dataset_dir**: Path to .mat data files
- **image_size**: Output resolution (128×128 pixels)
- **input_dim**: Boundary image channels (3 = RGB)
- **with_house**: Whether to include house-level features (disabled)
- **pos_dim**: Position attribute dimensions (25D)
- **area_dim**: Area attribute dimensions (10D)
- **Total attributes**: 25 + 10 = 35D

### Dataloader Parameters (Lines 47-50)
```python
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--train_shuffle', default='1', type=bool_flag)
```
- **batch_size=20**: 20 floor plans per batch
- **workers=8**: 8 parallel data loading processes
- **train_shuffle=True**: Randomize training data order

### Model Architecture Parameters (Lines 52-65)
```python
# architecture
parser.add_argument('--gene_layout', default='1', type=bool_flag)
parser.add_argument('--box_refine', default='1', type=bool_flag)
# input
parser.add_argument('--embedding_dim', default=128,type=int)
# refine
parser.add_argument('--refinement_dims', default='1024, 512, 256, 128, 64',type=int_tuple)
# box refine - now auto-detects num_objs from vocabulary
parser.add_argument('--box_refine_arch', default=None,type=str)
parser.add_argument('--roi_cat_feature',default='1',type=bool_flag)
# control
parser.add_argument('--gt_box', default=0, type=bool_flag)
parser.add_argument('--relative', default=1, type=bool_flag)
```
- **gene_layout**: Enable layout generation (refinement_net)
- **box_refine**: Enable box refinement (two-stage)
- **embedding_dim**: Room embedding size (128D)
- **refinement_dims**: Layer sizes for refinement CNN (unused in current implementation)
- **box_refine_arch**: CNN architecture string (set dynamically)
- **roi_cat_feature**: Concatenate room features with RoI features
- **gt_box**: Use ground truth boxes (for debugging)
- **relative**: Use relative coordinates (inside boundary)

### Loss Function Parameters (Lines 67-74)
```python
parser.add_argument('--mutex', default=1, type=bool_flag)
parser.add_argument('--inside', default=1, type=bool_flag)
parser.add_argument('--coverage', default=1, type=bool_flag)
parser.add_argument('--render', default=1, type=bool_flag)
parser.add_argument('--nsample', default=100,type=int)
parser.add_argument('--loss_refine', default=0, type=bool_flag)
parser.add_argument('--render_refine', default=0, type=bool_flag)
```
- **mutex**: Enable mutual exclusion loss (prevent overlap)
- **inside**: Enable containment loss (rooms inside boundary)
- **coverage**: Enable coverage loss (cover required areas)
- **render**: Enable rendering loss (box shape matching)
- **nsample**: Sample points for geometric losses (100 points)
- **loss_refine**: Apply geometric losses to refined boxes (disabled)
- **render_refine**: Apply render loss to refined boxes (disabled)

### Optimizer Parameters (Lines 76-83)
```python
parser.add_argument('--optimizer',default='Adam',type=str)
parser.add_argument('--scheduler',default='plateau',type=str)
parser.add_argument('--learning_rate', default=5e-5, type=float)  # Reduced from 1e-4
parser.add_argument('--decay_rate', default=1e-4, type=float)
parser.add_argument('--step_size', default=10, type=float)
parser.add_argument('--step_rate', default=0.5, type=float)
parser.add_argument('--grad_clip', default=1.0, type=float)  # Gradient clipping max norm
```
- **optimizer='Adam'**: Adam optimizer (also supports SGD, AdamW)
- **scheduler='plateau'**: ReduceLROnPlateau (reduce LR when metrics plateau)
- **learning_rate=5e-5**: Conservative LR to prevent NaN (reduced from 1e-4)
- **decay_rate=1e-4**: Weight decay (L2 regularization)
- **step_size=10**: Patience for scheduler (wait 10 epochs before reducing LR)
- **step_rate=0.5**: LR reduction factor (multiply by 0.5)
- **grad_clip=1.0**: Maximum gradient norm (prevent exploding gradients)

### Checkpoint Parameters (Lines 85-89)
```python
parser.add_argument('--save_interval', default=5, type=int)
parser.add_argument('--n_saved', default=20, type=int)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--skip_train', default=0, type=bool_flag)
```
- **save_interval=5**: Save checkpoint every 5 epochs
- **n_saved=20**: Keep last 20 checkpoints
- **pretrain**: Path to pretrained model (for resuming)
- **skip_train**: Skip training, only test (for evaluation)

### Training Parameters (Lines 91-94)
```python
parser.add_argument('--seed', default=74269,type=int)
parser.add_argument('--epoch', default=101,type=int)
parser.add_argument('--start_epoch',default=None,type=int)
```
- **seed=74269**: Random seed for reproducibility
- **epoch=101**: Total training epochs
- **start_epoch**: Resume from specific epoch

### Debug Parameters (Lines 96-101)
```python
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--multi_gpu', default=None, type=str)
parser.add_argument('--suffix',default=None,type=str)
parser.add_argument('--debug', default=0, type=bool_flag)
parser.add_argument('--test', default=0, type=bool_flag)
```
- **gpu='0'**: Which GPU to use
- **multi_gpu**: Multiple GPU string (e.g., '0,1,2,3')
- **suffix**: Custom experiment name suffix
- **debug**: Debug mode (uses validation set for training, only 6 epochs)
- **test**: Test mode flag

---

# Part 2: Setup Functions (Lines 105-197)

## Lines 105-109: check_manual_seed() - Set Random Seeds
```python
def check_manual_seed(args):
    seed = args.seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```
**Purpose**: Ensure reproducible results by seeding all random number generators.

**Note**: Line 242 comments this out (`# check_manual_seed(args)`), so seeds are NOT actually set!

---

## Lines 111-118: get_model() - Create Model Instance
```python
def get_model(args):
    return Model(embedding_dim=args.embedding_dim,
    image_size=args.image_size,
    input_dim = args.input_dim,
    attribute_dim=args.pos_dim+args.area_dim,
    refinement_dims=args.refinement_dims if args.gene_layout else None,
    box_refine_arch=args.box_refine_arch if args.box_refine else None,
    roi_cat_feature=args.roi_cat_feature)
```
**Key Logic**:
- `refinement_dims=None` if gene_layout disabled → No layout generation
- `box_refine_arch=None` if box_refine disabled → No box refinement

---

## Lines 120-142: Data Loading Functions

### Lines 120-121: get_dataset() - Load .mat Dataset
```python
def get_dataset(args,split='valid'):
    return FloorPlanDataset(f'{args.dataset_dir}/data_{split}.mat')
```
Loads: `data_train.mat`, `data_valid.mat`, or `data_test.mat`

### Lines 123-132: get_dataloader() - Create DataLoader
```python
def get_dataloader(args,dataset,split):
    print(f"{split},shuffle:",split=='train' and args.train_shuffle and (not args.debug))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if split=='train' and args.train_shuffle and (not args.debug) else False,
        num_workers=args.workers,
        drop_last=True if split=='train' else False,
        collate_fn=floorplan_collate_fn
    )
```
**Key Logic**:
- **Shuffle**: Only for training (unless debug mode)
- **drop_last**: Drop incomplete last batch in training (ensures consistent batch size)
- **collate_fn**: Custom function to batch variable-size graphs

### Lines 134-142: get_data_loaders() - Create All Loaders
```python
def  get_data_loaders(args):
    train_dataset = get_dataset(args,'train' if not args.debug else 'valid') if not args.skip_train else None
    valid_dataset = get_dataset(args,'valid')
    test_dataset = get_dataset(args,'test')

    train_loader = get_dataloader(args,train_dataset,'train') if not args.skip_train else None
    valid_loader = get_dataloader(args,valid_dataset,'valid')
    test_loader = get_dataloader(args,test_dataset,'test')
    return train_loader,valid_loader,test_loader
```
**Debug mode trick**: Uses validation set for training (faster iteration)

---

## Lines 144-161: get_optimizer() - Create Optimizer
```python
def get_optimizer(model,args):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = args.learning_rate,
            weight_decay=args.decay_rate
        )
    return optimizer
```
**Default**: Adam with LR=5e-5, weight_decay=1e-4

---

## Lines 163-168: get_scheduler() - Create Learning Rate Scheduler
```python
def get_scheduler(optimizer,args):
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_rate)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=args.step_rate,patience=args.step_size,threshold=0.005,verbose=True)
    return scheduler
```
**Default**: ReduceLROnPlateau
- **mode='max'**: Maximize metrics (IoU + accuracy)
- **factor=0.5**: Reduce LR by half
- **patience=10**: Wait 10 epochs before reducing
- **threshold=0.005**: Minimum improvement to reset patience

---

## Lines 170-197: get_losses() - Create Loss Functions

### Lines 172-176: Get Vocabulary and Setup
```python
loss = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get vocabulary to determine number of classes
vocab = get_vocab()
num_classes = len(vocab['object_idx_to_name'])
```
**num_classes=5**: After balcony removal (LivingRoom, MasterRoom, Kitchen, Bathroom, SecondBedroom)

### Lines 178-181: Create Class Weights
```python
# Create weight tensor for the actual number of classes (after balcony removal: 5 classes)
weight = torch.ones(num_classes).to(device)
# Note: No need to zero out External/ExteriorWall as they were removed with balconies
```
**Purpose**: Equal weighting for all room types (could be adjusted for class imbalance)

### Lines 183-196: Create Loss Modules
```python
if args.gene_layout:
    # Use ignore_index for background/boundary pixels that are outside valid room indices
    # Background is set to num_classes (5), boundary to num_classes+1 (6)
    loss['gene_ce'] = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=num_classes)
loss['box_mse'] = torch.nn.SmoothL1Loss()
if args.box_refine:
    loss['box_ref_mse'] = torch.nn.SmoothL1Loss()
if args.mutex:
    loss['mutex'] = MutexLoss(nsample=args.nsample)
if args.inside:
    loss['inside'] = InsideLoss(nsample=args.nsample)
if args.coverage:
    loss['coverage'] = CoverageLoss(nsample=args.nsample)
if args.render:
    loss['render'] = BoxRenderLoss(nsample=args.nsample)
return loss
```

**Loss Dictionary Contents**:
- `gene_ce`: Cross-entropy for layout (ignores background pixels with index 5)
- `box_mse`: Smooth L1 for initial boxes
- `box_ref_mse`: Smooth L1 for refined boxes
- `mutex`: Prevent room overlap (100 sample points)
- `inside`: Ensure rooms inside boundary (100 sample points)
- `coverage`: Ensure area coverage (100 sample points)
- `render`: Match box shapes (100 sample points)

---

## Lines 199-207: batch_cuda() - Move Batch to GPU
```python
def batch_cuda(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = list(batch)
    for i in range(len(batch)):
        if isinstance(batch[i],torch.Tensor):
            batch[i] = batch[i].to(device)
        elif isinstance(batch[i],list) and isinstance(batch[i][0],torch.Tensor):
            batch[i] = [e.to(device) for e in batch[i]]
    return batch
```
**Purpose**: Transfer all tensors in batch to GPU (or CPU if no GPU available)

---

# Part 3: Main Training Function (Lines 209-724)

## Lines 209-244: Setup and Initialization

### Lines 210-227: Create Experiment Directory
```python
args.epoch=args.epoch if not args.debug else 6
print("Create dir...")
start_date = str(datetime.datetime.now().strftime('%Y-%m-%d'))+("" if not args.debug else "_debug")+("" if not args.test else "_test")
if not os.path.exists(f'../experiment'):
    os.mkdir(f'../experiment')
experiment_dir = path.Path(f'../experiment/{start_date}')
experiment_dir.mkdir(exist_ok=True)
start_time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '' if args.suffix is None else args.suffix
file_dir = path.Path(f'{experiment_dir}/DeepLayout_{start_time}')
file_dir.mkdir(exist_ok=True)
checkpoints_dir = file_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
log_dir = file_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)
shutil.copy(__file__,log_dir/'train.py')
shutil.copytree('./model',log_dir/'model')
output_dir = file_dir.joinpath('output/')
output_dir.mkdir(exist_ok=True)
```

**Directory Structure**:
```
../experiment/
└── 2026-01-28/
    └── DeepLayout_2026-01-28_14-30-45/
        ├── checkpoints/    # Model checkpoints
        ├── logs/           # Copy of code, logs, tensorboard
        └── output/         # Test results
```

**Important**: Copies entire codebase to logs for reproducibility

### Lines 228-240: Setup Logging
```python
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(str(log_dir)+'/log.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
if args.skip_train:
    logger.info(f'python {args.argv}')
else:
    logger.info(f'python {args.argv} --skip_train 1 --pretrain ')
logger.info(args)
logger.info('---------------------------------------------------TRANING---------------------------------------------------')
logger.info(f'Use seed: {args.seed}')
```
**Logs**: All print statements and logging calls go to `logs/log.txt`

### Lines 244-265: Load Model and Data
```python
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else args.multi_gpu

print("Create dataloader...")
train_loader,valid_loader,test_loader = get_data_loaders(args)
print("Create model...")
model = get_model(args)
print("Gene:",model.refinement_net!=None and args.gene_layout)
print("Refine:",args.box_refine)
print("Cat feat:",args.roi_cat_feature)
print("GT BOX:",args.gt_box)
print("Iniside Loss:",args.inside)
print("Coverage Loss:",args.coverage)
print("Mutex Loss:",args.mutex)
print("Render Loss:",args.render)
logger.info(argparse.Namespace(embedding_dim=args.embedding_dim,
image_size=args.image_size,
input_dim = args.input_dim,
attribute_dim=args.pos_dim+args.area_dim,
refinement_dims=args.refinement_dims if args.gene_layout else None,
box_refine_arch=args.box_refine_arch if args.box_refine else None,
roi_cat_feature=args.roi_cat_feature))
logger.info(str(model))
```
**Logs model configuration** for debugging and reproducibility

### Lines 266-280: Create Optimizer and Load Checkpoint
```python
optimizer = get_optimizer(model,args)
scheduler = get_scheduler(optimizer,args)
loss = get_losses(args)

if args.pretrain is not None:
    # Load checkpoint with CPU/GPU compatibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.pretrain, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.pretrain} on {device}")

# Move model to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)
```
**map_location=device**: Ensures checkpoint loads correctly even if saved on different device

---

## Lines 282-294: NaN/Inf Detection Helper

```python
def check_tensor_for_nan(tensor, name, epoch, iteration):
    """Helper function to detect NaN/Inf in tensors and log details."""
    if tensor is None:
        return False
    if torch.isnan(tensor).any():
        logging.error(f"NaN detected in {name} at epoch {epoch}, iter {iteration}")
        logging.error(f"  Shape: {tensor.shape}, Min: {tensor[~torch.isnan(tensor)].min().item() if (~torch.isnan(tensor)).any() else 'all NaN'}, Max: {tensor[~torch.isnan(tensor)].max().item() if (~torch.isnan(tensor)).any() else 'all NaN'}")
        return True
    if torch.isinf(tensor).any():
        logging.error(f"Inf detected in {name} at epoch {epoch}, iter {iteration}")
        logging.error(f"  Shape: {tensor.shape}, Num Inf: {torch.isinf(tensor).sum().item()}")
        return True
    return False
```

**Purpose**: Comprehensive NaN/Inf detection with detailed logging for debugging

**Usage**: Called before model forward, after model forward, and throughout loss computation

---

## Lines 296-471: update() - Training Step Function ⭐⭐⭐

This is the **CORE TRAINING LOGIC**. Called once per batch.

### Lines 297-305: Learning Rate Warmup (EPOCH BEHAVIOR #1)

```python
def update(engine,batch):
    model.train()

    # Learning rate warmup for first 3 epochs to prevent early divergence
    epoch = engine.state.epoch
    if epoch <= 3:
        warmup_factor = min(1.0, epoch / 3.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate * warmup_factor
```

**Warmup Schedule**:
- **Epoch 1**: LR = 5e-5 × 1/3 = 1.67e-5 (33%)
- **Epoch 2**: LR = 5e-5 × 2/3 = 3.33e-5 (66%)
- **Epoch 3**: LR = 5e-5 × 3/3 = 5.0e-5 (100%)
- **Epoch 4+**: LR = 5e-5 (full speed)

**Why**: Prevent gradient explosion during early random initialization

---

### Lines 306-322: Input Validation and NaN Detection

```python
optimizer.zero_grad()

boundary,inside_box,objs,attrs,triples,layout,boxes,inside_coords,obj_to_img,triple_to_img,name = batch_cuda(batch)

if args.relative: boxes = box_rel2abs(boxes,inside_box,obj_to_img)

# CRITICAL: Check input data for NaN/Inf before model forward
iteration = engine.state.iteration
has_bad_input = False
has_bad_input |= check_tensor_for_nan(boundary, "boundary", epoch, iteration)
has_bad_input |= check_tensor_for_nan(inside_box, "inside_box", epoch, iteration)
has_bad_input |= check_tensor_for_nan(boxes, "boxes", epoch, iteration)
has_bad_input |= check_tensor_for_nan(attrs, "attrs", epoch, iteration)

if has_bad_input:
    logging.error(f"Skipping batch due to bad input data")
    return {'total_loss': 0.0}
```

**Safety**: Skip corrupted batches rather than crash

---

### Lines 324-346: Model Forward Pass with Epoch Control (EPOCH BEHAVIOR #2)

```python
model_out = model(
    objs,
    triples,
    boundary,
    obj_to_img = obj_to_img,
    attributes=attrs,
    boxes_gt= boxes if args.gt_box else None,
    generate = args.gene_layout,  # Generate from epoch 1 for gradual training
    refine = args.box_refine and engine.state.epoch>2,  # ⭐ CRITICAL LINE
    relative = args.relative,
    inside_box=inside_box if args.relative else None,
)
boxes_pred, gene_layout, boxes_refine = model_out

# CRITICAL: Check model outputs for NaN/Inf
has_bad_output = False
has_bad_output |= check_tensor_for_nan(boxes_pred, "boxes_pred", epoch, iteration)
has_bad_output |= check_tensor_for_nan(gene_layout, "gene_layout", epoch, iteration)
has_bad_output |= check_tensor_for_nan(boxes_refine, "boxes_refine", epoch, iteration)

if has_bad_output:
    logging.error(f"NaN/Inf in model output - skipping batch to prevent gradient corruption")
    return {'total_loss': 0.0}
```

**KEY LINE 332**: `refine = args.box_refine and engine.state.epoch>2`

**Model Behavior**:
- **Epochs 1-2**: `refine=False` → Single-stage (boxes_refine=None)
- **Epoch 3+**: `refine=True` → Two-stage (boxes_refine computed)

---

### Lines 348-355: Loss Weight Schedule (EPOCH BEHAVIOR #3)

```python
# Initialize total_loss as None, will be set to first valid loss
total_loss = None
loss_items = {}
epoch = engine.state.epoch
# Even more gradual step_weight progression to prevent NaN at epoch 7
# Extended to 31 epochs with smaller increments around the problematic epoch 7 range
# Epochs: 1,     2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31+
step_weight = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.50]
```

**The 31-Epoch Schedule**: Gradual increase of layout generation loss weight

**Why so gradual?**: Comment mentions "prevent NaN at epoch 7" - this was discovered through trial and error

---

### Lines 356-440: Loss Computation Loop

This iterates through all enabled losses and computes them with epoch-dependent behavior.

#### Lines 358-359: Box MSE Loss (Always Active)
```python
for name in loss:
    l = None
    if name=='box_mse':
        l = loss[name](boxes_pred,boxes)
```
**Weight**: 1.0 (no scaling)
**Purpose**: Main loss for box prediction

---

#### Lines 360-385: Gene_ce Loss (Gradual from Epoch 1) (EPOCH BEHAVIOR #4)

```python
elif name=='gene_ce':
    # Gene_ce now starts from epoch 1 with very small weight (0.005)
    # This ensures refinement_net gets gradients from the start

    # Skip if gene_layout wasn't generated
    if gene_layout is None:
        continue

    # Check for NaN in gene_layout before computing loss
    if torch.isnan(gene_layout).any() or torch.isinf(gene_layout).any():
        logging.error(f"NaN/Inf detected in gene_layout at epoch {epoch}")
        continue

    # Early warning: check for extreme values that might lead to NaN
    gene_max = gene_layout.abs().max().item()
    if gene_max > 50.0:
        logging.warning(f"Large gene_layout values detected at epoch {epoch}: max={gene_max:.2f}")

    # Use gradual step_weight starting from epoch 1 (21 epochs total)
    weight_idx = min(epoch-1, len(step_weight)-1)
    current_weight = step_weight[weight_idx]
    l = current_weight * loss[name](gene_layout,layout)

    # Log gene_ce contribution periodically
    if engine.state.iteration % 100 == 0:
        logging.info(f"Epoch {epoch}, gene_ce loss: {loss[name](gene_layout,layout).item():.4f}, weight: {current_weight:.2f}, contribution: {l.item():.4f}")
```

**Weight Schedule**:
- **Epoch 1**: 0.005 (0.5%)
- **Epoch 2**: 0.01 (1%)
- **Epoch 3**: 0.02 (2%)
- **Epoch 7**: 0.06 (6%) ← Previously caused NaN
- **Epoch 31+**: 0.50 (50%)

**Extra Safety**: Checks for extreme values (>50) that might lead to NaN

---

#### Lines 386-395: Mutex Loss (EPOCH BEHAVIOR #5)

```python
elif name=='mutex':
    l = 0.1*loss[name](boxes_pred,obj_to_img,objs)
    if torch.isnan(l) or torch.isinf(l):
        logging.warning(f"NaN/Inf in mutex loss at epoch {epoch}, skipping")
        l = None
    elif args.box_refine and args.loss_refine and epoch>2:  # ⭐ Epoch check
        l_refine = loss[name](boxes_refine,obj_to_img,objs)
        if not (torch.isnan(l_refine) or torch.isinf(l_refine)):
            l += l_refine
```

**Behavior**:
- **Epochs 1-2**: Only compute on `boxes_pred`
- **Epoch 3+**: Compute on BOTH `boxes_pred` AND `boxes_refine` (if loss_refine enabled)

**Weight**: 0.1 (10% of base)

---

#### Lines 396-403: Inside Loss (Same Pattern)

```python
elif name=='inside':
    l = 0.1*loss[name](boxes_pred,inside_box,obj_to_img)
    if torch.isnan(l) or torch.isinf(l):
        logging.warning(f"NaN/Inf in inside loss at epoch {epoch}, skipping")
        l = None
    elif args.box_refine and args.loss_refine and epoch>2:  # ⭐ Epoch check
        l_refine = loss[name](boxes_refine,inside_box,obj_to_img)
        if not (torch.isnan(l_refine) or torch.isinf(l_refine)):
            l += l_refine
```
**Same pattern as mutex**: Only add refinement component after epoch 2

---

#### Lines 404-412: Coverage Loss (Same Pattern)

```python
elif name=='coverage':
    l = 0.1*loss[name](boxes_pred,inside_coords,obj_to_img)
    if torch.isnan(l) or torch.isinf(l):
        logging.warning(f"NaN/Inf in coverage loss at epoch {epoch}, skipping")
        l = None
    elif args.box_refine and args.loss_refine and epoch>2:  # ⭐ Epoch check
        l_refine = loss[name](boxes_refine,inside_coords,obj_to_img)
        if not (torch.isnan(l_refine) or torch.isinf(l_refine)):
            l += l_refine
```

---

#### Lines 413-421: Render Loss (Same Pattern)

```python
elif name=='render':
    l = loss[name](boxes_pred,boxes)
    if torch.isnan(l) or torch.isinf(l):
        logging.warning(f"NaN/Inf in render loss at epoch {epoch}, skipping")
        l = None
    elif args.box_refine and args.loss_refine and epoch>2:  # ⭐ Epoch check
        l_refine = loss[name](boxes_refine,boxes)
        if not (torch.isnan(l_refine) or torch.isinf(l_refine)):
            l += l_refine
```

---

#### Lines 422-429: Box Refinement MSE Loss (EPOCH BEHAVIOR #6)

```python
elif name=='box_ref_mse' and epoch>2:  # ⭐ Only starts at epoch 3
    # Start box_ref_mse from epoch 3, use same gradual weights
    # Map to step_weight starting from epoch 3 (index 2 in the array)
    weight_idx = min(epoch-1, len(step_weight)-1)  # Use same progression as gene_ce
    l = step_weight[weight_idx]*loss[name](boxes_refine,boxes)
    if torch.isnan(l) or torch.isinf(l):
        logging.warning(f"NaN/Inf in box_ref_mse loss at epoch {epoch}, skipping")
        l = None
```

**Behavior**:
- **Epochs 1-2**: Completely disabled (boxes_refine doesn't exist)
- **Epoch 3**: Weight = 0.02 (2%)
- **Epoch 31+**: Weight = 0.50 (50%)

**Same schedule as gene_ce**: Both ramp up together

---

#### Lines 431-440: Accumulate Losses

```python
if l is not None:
    # Final safety check before adding to total
    if torch.isnan(l) or torch.isinf(l):
        logging.warning(f"NaN/Inf detected in {name} loss, skipping")
    else:
        if total_loss is None:
            total_loss = l
        else:
            total_loss = total_loss + l
        loss_items[name]=l.item()

# If all losses were None/NaN, create a zero tensor for backward
if total_loss is None:
    device = boxes_pred.device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    logging.warning(f"All losses were None/NaN at epoch {epoch}, using zero loss")

loss_items['total_loss'] = total_loss.item()
```

**Robust accumulation**: Each loss checked individually, skip if NaN

---

### Lines 450-470: Backward Pass and Gradient Clipping

```python
# Check for NaN before backward pass
if torch.isnan(total_loss) or torch.isinf(total_loss):
    logging.error(f"NaN or Inf loss detected at epoch {epoch}, batch {engine.state.iteration}. Loss items: {loss_items}")
    logging.error(f"Skipping this batch to prevent gradient corruption.")
    return loss_items

total_loss.backward()

# Compute gradient norm before clipping for monitoring
# Note: clip_grad_norm_ returns the UNCLIPPED total norm, but DOES apply clipping
# So logged grad_norm is pre-clipping (for diagnosis), but gradients ARE clipped
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

# Log gradient norm periodically (every 100 iterations) for diagnosis
# If grad_norm >> max_norm, the network is under stress even with clipping active
if engine.state.iteration % 100 == 0:
    clip_ratio = total_norm / args.grad_clip if total_norm > 0 else 0
    logging.info(f"Epoch {epoch}, Iter {engine.state.iteration}: grad_norm={total_norm:.4f} (clip_ratio={clip_ratio:.1f}x), loss={total_loss.item():.6f}")

optimizer.step()
return loss_items
```

**Gradient Clipping**:
- **max_norm=1.0**: Gradients scaled down if norm exceeds 1.0
- **Logging**: Tracks gradient norm every 100 iterations
- **clip_ratio**: Shows how much clipping is needed (>1.0 means clipping active)

**Purpose**: Prevent exploding gradients from destabilizing training

---

## Lines 473-609: inference() - Validation Step Function

This is similar to `update()` but:
- No gradient computation (`with torch.no_grad()`)
- No optimizer steps
- Returns predictions for metric computation
- Different epoch gating logic

### Lines 473-491: Setup and Input Validation

```python
def inference(engine,batch):
    model.eval()
    with torch.no_grad():
        boundary,inside_box,objs,attrs,triples,layout,boxes,inside_coords,obj_to_img,triple_to_img,name = batch_cuda(batch)

        # CRITICAL: Validate room indices in validation data
        if (objs >= 5).any() or (objs < 0).any():
            logging.warning(f"Invalid room indices in validation batch objs: {objs[objs >= 5].tolist() if (objs >= 5).any() else 'none'}")
            logging.warning(f"Batch name: {name}")
            # Skip this batch
            return {'total_loss': 0.0}

        # CRITICAL: Validate layout tensor
        if (layout > 5).any() or (layout < 0).any():
            invalid_vals = layout[(layout > 5) | (layout < 0)].unique().tolist()
            logging.warning(f"Invalid layout indices in validation batch: {invalid_vals}")
            logging.warning(f"Batch name: {name}")
            # Clamp layout to valid range
            layout = torch.clamp(layout, 0, 5)
```

**Extra validation checks**: Catches data with old vocabulary (e.g., balcony indices)

---

### Lines 493-507: Model Forward (Always Full Model in Validation)

```python
if args.relative: boxes = box_rel2abs(boxes,inside_box,obj_to_img)

model_out = model(
    objs,
    triples,
    boundary,
    obj_to_img = obj_to_img,
    attributes=attrs,
    boxes_gt= boxes if args.gt_box else None,
    generate = args.gene_layout,
    refine = args.box_refine,  # ⭐ No epoch check! Always use full model
    relative = args.relative,
    inside_box=inside_box if args.relative else None,
)
boxes_pred, gene_layout, boxes_refine = model_out
```

**Important difference**: `refine=args.box_refine` (no epoch check)

**Validation always uses full model** (even epochs 1-2) for consistent comparison

---

### Lines 509-576: Loss Computation (With Epoch Gating) (EPOCH BEHAVIOR #7)

```python
# Initialize total_loss as None, will be set to first valid loss
total_loss = None
loss_items = {}
for name in loss:
    l = None
    if name=='box_mse':
        l = loss[name](boxes_pred,boxes)
    if engine.state.epoch>1:  # ⭐ Most losses only after epoch 1
        if name=='gene_ce':
            # Check for NaN in gene_layout before computing loss (validation)
            if torch.isnan(gene_layout).any() or torch.isinf(gene_layout).any():
                logging.warning(f"NaN/Inf detected in gene_layout during validation at epoch {engine.state.epoch}")
                l = None
            else:
                l = loss[name](gene_layout,layout)
        elif name=='mutex':
            l = 0.1*loss[name](boxes_pred,obj_to_img,objs)
            if torch.isnan(l) or torch.isinf(l):
                l = None
            elif args.box_refine and args.loss_refine:
                l_add = 0.1*loss[name](boxes_refine,obj_to_img,objs)
                if not (torch.isnan(l_add) or torch.isinf(l_add)):
                    l += l_add
        # ... similar for inside, coverage, render ...

    if engine.state.epoch>2:  # ⭐ Box refinement loss only after epoch 2
        if name=='box_ref_mse':
            l = loss[name](boxes_refine,boxes)
            if torch.isnan(l) or torch.isinf(l):
                l = None
```

**Validation Loss Gating**:
- **Epoch 1**: Only `box_mse`
- **Epoch 2+**: Add `gene_ce`, `mutex`, `inside`, `coverage`, `render`
- **Epoch 3+**: Add `box_ref_mse`

**Note**: No gradual weighting in validation - losses are at full strength

---

### Lines 586-609: Prepare Outputs for Metrics

```python
# boxes pred
boxes_pred = boxes_pred.detach()
boxes_pred = centers_to_extents(boxes_pred)

if args.gene_layout:
    gene_layout = gene_layout*boundary[:,:1]

# boxes refine
if args.box_refine:
    boxes_refine = boxes_refine.detach()
    boxes_refine = centers_to_extents(boxes_refine)

# gt
boxes = centers_to_extents(boxes)

return {
    'loss':loss_items,
    'pred':[
        boxes_pred,
        gene_layout.detach() if args.gene_layout else None,
        boxes_refine if args.box_refine else None,
        ],
    'gt':[layout,boxes]
}
```

**Format conversion**: Center format → Extent format for IoU computation

**Return format**: Structured dict for metric computation

---

## Lines 611-630: Engine Setup

### Lines 611-620: Create Engines and Set Start Epoch
```python
print("Create trainer...")
optimizer.step()
scheduler.step(0)
trainer = Engine(update)  # type: ignore
valid_evaluator = Engine(inference)  # type: ignore

if args.start_epoch is not None:
    @trainer.on(Events.STARTED)  # type: ignore
    def set_up_state(engine):
        engine.state.epoch = args.start_epoch
```

**PyTorch Ignite**: Uses Engine abstraction for training loop

**Initialization trick**: `optimizer.step()` and `scheduler.step(0)` called once to initialize

---

### Lines 622-630: Scheduler Callback

```python
total_func = lambda e:(e.state.metrics['box_iou']+(e.state.metrics['gene_acc'] if args.gene_layout else 0)+(e.state.metrics['box_refine_iou'] if args.box_refine else 0))

@valid_evaluator.on(Events.COMPLETED)  # type: ignore
def schedual(engine):
    optimizer.step()
    if args.scheduler == 'step':
        scheduler.step()
    else:
        scheduler.step(total_func(engine))
```

**Metric for scheduler**: Sum of box_iou + gene_acc + box_refine_iou

**Goal**: Maximize combined performance

---

## Lines 632-669: Validation Callback (Currently Disabled!)

```python
@trainer.on(Events.EPOCH_COMPLETED)  # type: ignore
def evaluate(engine):
    # TEMPORARY FIX: Skip validation because data_valid.mat has old vocabulary
    # TODO: Regenerate data_valid.mat with balconies removed (matching data_train.mat from Jan 19)
    logging.info(f"Epoch {engine.state.epoch} completed - SKIPPING validation (data_valid.mat has old vocabulary)")

    # Original validation code (commented out until data_valid.mat is regenerated):
    """
    # Run validation only every 5 epochs to save time
    if engine.state.epoch % 5 == 0 or engine.state.epoch == 1:
        logging.info(f"Running validation at epoch {engine.state.epoch}")
        try:
            # Clear CUDA cache before validation to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            valid_evaluator.run(valid_loader)
            # ... more code ...
```

**CRITICAL ISSUE**: Validation is completely disabled!

**Reason**: `data_valid.mat` has old vocabulary (includes balcony indices)

**Original behavior**: Run validation every 5 epochs (to save time)

---

## Lines 671-706: Metrics and Logging

### Lines 671-676: Attach Metrics to Validator
```python
# Metrics
MetricAverage(output_transform=lambda output:iou(output['pred'][0],output['gt'][1])).attach(valid_evaluator,'box_iou')
if args.gene_layout:
    MetricAverage(output_transform=lambda output:image_acc_ignore(output['pred'][1],output['gt'][0],13)).attach(valid_evaluator,'gene_acc')
if args.box_refine:
    MetricAverage(output_transform=lambda output:iou(output['pred'][2],output['gt'][1])).attach(valid_evaluator,'box_refine_iou')
```

**Metrics computed**:
- **box_iou**: IoU for initial boxes
- **gene_acc**: Layout accuracy (ignoring background index 13)
- **box_refine_iou**: IoU for refined boxes

---

### Lines 680-694: Progress Bars and TensorBoard

```python
# TQDM
ProgressBar(persist=True).attach(trainer, output_transform=lambda o:{'loss':o['total_loss']}, metric_names='all')  # type: ignore
ProgressBar(persist=False).attach(valid_evaluator, output_transform=lambda o:{'loss':o['loss']['total_loss']},metric_names='all')  # type: ignore

# Tensorboard
tb_logger = TensorboardLogger(log_dir=log_dir)  # type: ignore
tb_logger.attach(trainer,
             log_handler=OutputHandler(tag="train",output_transform=lambda o: o,metric_names='all'),  # type: ignore
             event_name=Events.ITERATION_COMPLETED)  # type: ignore
tb_logger.attach(trainer,
             log_handler=OptimizerParamsHandler(optimizer),  # type: ignore
             event_name=Events.ITERATION_STARTED)  # type: ignore
tb_logger.attach(valid_evaluator,
             log_handler=OutputHandler(tag="valid",output_transform=lambda o:o['loss'],metric_names='all', global_step_transform=global_step_from_engine(trainer)),  # type: ignore
             event_name=Events.EPOCH_COMPLETED)  # type: ignore
```

**TensorBoard logs**:
- Training loss every iteration
- Learning rate every iteration
- Validation metrics every epoch

**View with**: `tensorboard --logdir=logs/`

---

### Lines 696-706: Text Logging Callbacks

```python
# Logging
@trainer.on(Events.EPOCH_COMPLETED)  # type: ignore
def log_results(engine):
    logging.info(f'Train, Epoch{engine.state.epoch}, Loss: {str(engine.state.output)}')

@valid_evaluator.on(Events.EPOCH_COMPLETED)  # type: ignore
def log_results(engine):
    loss = engine.state.output['loss']
    metrics = engine.state.metrics
    logging.info(f'Valid, Epoch{engine.state.epoch}, Loss: {str(loss)}')
    logging.info(f'Valid, Epoch{engine.state.epoch}, Metrics: {str(metrics)}')
```

**Logs to**: `logs/log.txt`

---

## Lines 708-723: Checkpointing

```python
# Checkpoint - save_interval moved to event handler attachment
epoch_saver = ModelCheckpoint(checkpoints_dir, 'epoch', n_saved=args.n_saved, require_empty=False, create_dir=True)  # type: ignore
latest_saver = ModelCheckpoint(checkpoints_dir, 'latest', score_function=lambda e:e.state.epoch, n_saved=1, require_empty=False, create_dir=True)  # type: ignore
loss_saver = ModelCheckpoint(checkpoints_dir, 'loss', score_function=lambda e:-e.state.output['loss']['total_loss'], n_saved=1, require_empty=False, create_dir=True)  # type: ignore

trainer.add_event_handler(Events.EPOCH_COMPLETED, latest_saver, {'model': model,'opt':optimizer})  # type: ignore
# Use Events.EPOCH_COMPLETED(every=N) for save_interval
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=args.save_interval), epoch_saver, {'model': model,'opt':optimizer})  # type: ignore
# Changed: Attach loss_saver to trainer instead of valid_evaluator since validation is skipped
# This saves the best training loss checkpoint instead of validation loss
trainer.add_event_handler(Events.EPOCH_COMPLETED, loss_saver, {'model': model})  # type: ignore
# NOTE: When validation is re-enabled, uncomment the line below and remove the line above
# valid_evaluator.add_event_handler(Events.COMPLETED, loss_saver, {'model': model})  # type: ignore
```

**Three checkpoint types**:

1. **latest_saver**: Saves most recent model (every epoch)
   - File: `latest_model=X.pt` (X = epoch number)
   - Keeps: 1 checkpoint

2. **epoch_saver**: Periodic saves
   - File: `epoch_model=X.pt`
   - Keeps: 20 checkpoints (n_saved=20)
   - Saves: Every 5 epochs (save_interval=5)

3. **loss_saver**: Best model by loss
   - File: `loss_model=X.pt`
   - Keeps: 1 checkpoint (best only)
   - **Currently**: Uses training loss (validation disabled)
   - **Should use**: Validation loss (when fixed)

---

### Lines 722-724: Start Training!

```python
if not args.skip_train:
    trainer.run(train_loader,max_epochs=args.epoch)
tb_logger.close()
```

**This is where training actually runs** (if not skip_train)

**trainer.run()**: Loops through epochs and batches, calling `update()` each time

---

# Part 4: Testing Phase (Lines 726-868)

After training completes, run inference on test set and save detailed results.

## Lines 726-843: test() - Test Function

Very similar to `inference()` but:
- Saves detailed per-sample outputs
- Converts predictions to final format
- Computes per-sample metrics

### Lines 730-748: Input Validation (Same as inference)

```python
def test(engine,batch):
    model.eval()
    with torch.no_grad():
        boundary,inside_box,objs,attrs,triples,layout,boxes,inside_coords,obj_to_img,triple_to_img,name = batch_cuda(batch)

        # CRITICAL: Log room indices to diagnose out-of-bounds issue
        if (objs >= 5).any() or (objs < 0).any():
            logging.error(f"Invalid room indices in test batch objs: {objs[objs >= 5].tolist() if (objs >= 5).any() else 'none'}")
            logging.error(f"Batch name: {name}")
            logging.error(f"All objs: {objs.tolist()}")
            # Skip this batch
            return {}

        # CRITICAL: Also check layout tensor (segmentation map)
        if (layout > 5).any() or (layout < 0).any():
            invalid_vals = layout[(layout > 5) | (layout < 0)].unique().tolist()
            logging.error(f"Invalid layout indices in test batch: {invalid_vals}")
            logging.error(f"Batch name: {name}")
            logging.error(f"Layout min: {layout.min().item()}, max: {layout.max().item()}")
            # Clamp layout to valid range [0, 5]
            layout = torch.clamp(layout, 0, 5)
```

---

### Lines 750-776: Model Forward and Format Conversion

```python
model_out = model(
    objs,
    triples,
    boundary,
    obj_to_img = obj_to_img,
    attributes=attrs,
    boxes_gt= boxes if args.gt_box else None,
    generate = args.gene_layout,
    refine = args.box_refine,
    relative = args.relative,
    inside_box=inside_box if args.relative else None,
)
boxes_pred, gene_layout, boxes_refine = model_out

''' box: x_c,y_c,w,h -> x0,y0,x1,y1 '''
# boxes pred
boxes_pred = boxes_pred.detach()
boxes_pred = centers_to_extents(boxes_pred)

# boxes refine
if args.box_refine:
    boxes_refine = boxes_refine.detach()
    boxes_refine = centers_to_extents(boxes_refine)

# gt
if args.relative: boxes = box_rel2abs(boxes,inside_box,obj_to_img)
boxes = centers_to_extents(boxes)
```

**Format**: Center format → Extent format [x0, y0, x1, y1]

---

### Lines 778-800: Process Layout Predictions

```python
''' layout: B*C*H*W->B*H*W '''
if args.gene_layout:
    gene_layout = gene_layout*boundary[:,:1]
    gene_preds = torch.argmax(gene_layout.softmax(1).detach(),dim=1)

''' layout with outside'''
for i in range(len(layout)):
    mask = boundary[i,0]==0
    if args.gene_layout:
        gene_preds[i][mask]=13

''' mertics '''
# box iou
box_ious = iou(boxes_pred,boxes)
box_refine_ious = None
if args.box_refine:
    box_refine_ious = iou(boxes_refine,boxes)

gene_acc_all = None
gene_acc_fg = None
if args.gene_layout:
    gene_acc_all = image_acc(gene_preds,layout)
    gene_acc_fg = image_acc_ignore(gene_preds,layout,13)
```

**Processing**:
1. Mask layout with boundary (zero out exterior)
2. Argmax to get predicted class per pixel
3. Set exterior pixels to index 13 (outside)
4. Compute per-sample IoU and accuracy

---

### Lines 802-835: Save Per-Sample Results

```python
''' save output '''
for i in range(len(layout)):
    ''' objs '''
    obj = objs[obj_to_img==i].cpu().numpy()

    ''' box '''
    box_pred = boxes_pred[obj_to_img==i]
    box_pred = box_pred.cpu().numpy()
    box_iou = box_ious[obj_to_img==i].view(-1).cpu().numpy()

    box_refine = None
    if args.box_refine:
        box_refine = boxes_refine[obj_to_img==i].cpu().numpy()
        box_refine_iou = box_refine_ious[obj_to_img==i].view(-1).cpu().numpy()

    ''' layout '''
    if args.gene_layout:
        gene_pred = gene_preds[i].cpu().numpy().astype('uint8')


    output[name[i]] = {
            'obj':obj,
            'box_gt':boxes[obj_to_img==i].cpu().numpy(),

            'box_pred':box_pred,
            'box_iou':box_iou,

            'box_refine':box_refine if args.box_refine else None,
            'box_refine_iou':box_refine_iou if args.box_refine else None,

            'gene_pred':gene_pred if args.gene_layout else None,
            'gene_acc_all': gene_acc_all[i].item() if args.gene_layout else None,
            'gene_acc_fg':gene_acc_fg[i].item() if args.gene_layout else None
            }
```

**Output dict structure** (per sample):
```python
{
    'sample_name_001': {
        'obj': [0, 1, 2, 3],  # Room types
        'box_gt': [[x0,y0,x1,y1], ...],  # Ground truth boxes
        'box_pred': [[x0,y0,x1,y1], ...],  # Predicted boxes
        'box_iou': [0.75, 0.82, 0.68, 0.91],  # Per-room IoU
        'box_refine': [[x0,y0,x1,y1], ...],  # Refined boxes
        'box_refine_iou': [0.78, 0.85, 0.72, 0.93],  # Per-room refined IoU
        'gene_pred': (128, 128),  # Layout prediction (uint8 image)
        'gene_acc_all': 0.87,  # Overall accuracy
        'gene_acc_fg': 0.92  # Foreground accuracy
    },
    'sample_name_002': { ... },
    ...
}
```

---

## Lines 845-868: Test Evaluation and Saving

### Lines 845-865: Run Test and Save Results
```python
test_evaluator = Engine(test)  # type: ignore

MetricAverage(output_transform=lambda output:iou(output['pred'][0],output['gt'][1])).attach(test_evaluator,'box_iou')

if args.gene_layout:
    MetricAverage(output_transform=lambda output:image_acc_ignore(output['pred'][1],output['gt'][0],13)).attach(test_evaluator,'gene_acc')
    MetricAverage(output_transform=lambda output:image_acc(output['pred'][1],output['gt'][0])).attach(test_evaluator,'gene_acc_all')
if args.box_refine:
    MetricAverage(output_transform=lambda output:iou(output['pred'][2],output['gt'][1])).attach(test_evaluator,'box_refine_iou')

ProgressBar(persist=False).attach(test_evaluator)  # type: ignore
@test_evaluator.on(Events.COMPLETED)  # type: ignore
def save_metrics(engine):
    metrics = engine.state.metrics
    with open(f'{output_dir}/output_{start_time}_metrics.json','w') as f:
        f.write(str(metrics))

if not args.skip_train:
    test_evaluator.run(valid_loader)
else:
    test_evaluator.run(test_loader)
with open(f'{output_dir}/output_{start_time}.pkl','wb') as f:
    pickle.dump(output,f,pickle.HIGHEST_PROTOCOL)
```

**Behavior**:
- If training was run: Test on validation set
- If skip_train: Test on test set (for final evaluation)

**Saves two files**:
1. `output_TIMESTAMP_metrics.json`: Aggregate metrics
2. `output_TIMESTAMP.pkl`: Detailed per-sample results (Python pickle)

---

# Part 5: Entry Point (Lines 869-872)

```python
if __name__ == "__main__":
    args = parse_args()
    args.argv = ' '.join(sys.argv)
    main(args)
```

**Standard Python entry point**

**args.argv**: Saves command line for logging (reproducibility)

---

# Summary: Complete Epoch Behavior

## Epoch-by-Epoch Breakdown

| Epoch | LR | Model | gene_ce | box_ref_mse | Refine Losses | Val Losses |
|-------|----|----|---------|-------------|---------------|------------|
| **1** | 33% | Single | 0.5% | - | - | box_mse only |
| **2** | 66% | Single | 1% | - | - | All except box_ref |
| **3** | 100% | **Two-stage** | 2% | 2% | ✓ Added | All losses |
| **4-30** | 100% | Two-stage | 3%→46% | 3%→46% | ✓ Active | All losses |
| **31+** | 100% | Two-stage | 50% | 50% | ✓ Active | All losses |

---

## All Epoch-Dependent Behaviors

### 1. Learning Rate Warmup (lines 299-304)
```python
if epoch <= 3:
    warmup_factor = min(1.0, epoch / 3.0)
    lr = args.learning_rate * warmup_factor
```

### 2. Box Refinement Activation (line 332)
```python
refine = args.box_refine and engine.state.epoch>2
```

### 3. Gene_ce Gradual Weighting (lines 355, 361-385)
```python
step_weight = [0.005, 0.01, 0.02, ...]
weight_idx = min(epoch-1, len(step_weight)-1)
l = step_weight[weight_idx] * loss['gene_ce'](gene_layout, layout)
```

### 4. Box_ref_mse Activation (line 422)
```python
elif name=='box_ref_mse' and epoch>2:
    weight_idx = min(epoch-1, len(step_weight)-1)
    l = step_weight[weight_idx] * loss[name](boxes_refine, boxes)
```

### 5. Refinement in Geometric Losses (lines 391, 400, 409, 418)
```python
elif args.box_refine and args.loss_refine and epoch>2:
    l += loss[name](boxes_refine, ...)
```

### 6. Validation Loss Gating (line 516)
```python
if engine.state.epoch>1:
    # Compute gene_ce, mutex, inside, coverage, render
```

### 7. Validation Box_ref_mse (line 557)
```python
if engine.state.epoch>2:
    if name=='box_ref_mse':
        l = loss[name](boxes_refine, boxes)
```

---

# Training Flow Diagram

```
Program Start
    ↓
Parse Arguments (lines 36-103)
    ↓
Create Directories & Logging (lines 210-240)
    ↓
Load Data (lines 247)
    ↓
Create Model, Optimizer, Losses (lines 248-268)
    ↓
Load Checkpoint (if pretrain) (lines 270-275)
    ↓
Move to GPU (lines 277-280)
    ↓
Define update() function (lines 296-471)
    │
    ├─ Check epoch, adjust LR (lines 299-304)
    ├─ Forward with epoch-dependent refine flag (line 332)
    ├─ Compute losses with epoch-dependent weights (lines 356-440)
    ├─ Backward + gradient clipping (lines 450-470)
    └─ Return loss_items
    ↓
Define inference() function (lines 473-609)
    ↓
Attach metrics, logging, checkpointing (lines 671-720)
    ↓
trainer.run(train_loader, max_epochs=101) ← TRAINING LOOP
    │
    ├─ For each epoch:
    │   ├─ For each batch:
    │   │   └─ Call update() → forward + backward + step
    │   └─ evaluate() callback → validation (CURRENTLY DISABLED)
    │
    └─ After all epochs, training done
    ↓
test_evaluator.run(test_loader) ← TESTING
    │
    ├─ For each batch:
    │   └─ Call test() → forward + save results
    │
    └─ Save metrics and detailed results to .pkl
    ↓
Program End
```

---

# Key Insights

## 1. Why Training is Unstable

**Evidence**:
- 31-epoch gradual loss weighting (not typical)
- NaN/Inf checks everywhere (lines 282-294, 312-346, 388-429)
- Conservative learning rate (5e-5 instead of 1e-4)
- Validation disabled due to data issues
- Extensive logging and safety checks

**Root cause**: Layout generation CNN (refinement_net) produces unstable gradients

**Solution strategy**: Gradual activation prevents sudden gradient spikes

---

## 2. The Critical Epoch: Epoch 3

**What changes**:
- Learning rate reaches 100%
- Two-stage model activates (box refinement begins)
- Box_ref_mse loss turns on
- Refinement added to geometric losses

**This is when the model becomes "complete"**

---

## 3. Validation is Broken

**Line 636**: Validation completely disabled

**Reason**: `data_valid.mat` has old vocabulary (includes removed balconies)

**Impact**: Cannot properly evaluate model during training!

**Fix needed**: Regenerate `data_valid.mat` with matching vocabulary

---

## 4. Training vs Inference Differences

**Training (epochs 1-2)**:
```python
refine = args.box_refine and engine.state.epoch>2  # False
```

**Validation (all epochs)**:
```python
refine = args.box_refine  # True
```

**Testing**:
```python
refine = args.box_refine  # True (always full model)
```

**Insight**: Evaluation always uses full model for fair comparison

---

# Common Usage Patterns

## Train from Scratch
```bash
python train.py --dataset_dir ./data --epoch 101 --gpu 0
```

## Resume Training
```bash
python train.py --pretrain ../experiment/.../checkpoints/latest_model_50.pt --start_epoch 51
```

## Evaluation Only
```bash
python train.py --skip_train 1 --pretrain ../experiment/.../checkpoints/loss_model_best.pt
```

## Debug Mode
```bash
python train.py --debug 1 --epoch 6
```
- Uses validation set for training (faster)
- Only 6 epochs
- Appends "_debug" to experiment name

## Test with Ground Truth Boxes
```bash
python train.py --gt_box 1 --skip_train 1 --pretrain ...
```
- Uses ground truth boxes (oracle experiment)
- Tests upper bound of layout generation

---

# Files Generated

## During Training

**Directory structure**:
```
../experiment/2026-01-28/DeepLayout_2026-01-28_14-30-45/
├── checkpoints/
│   ├── latest_model_1.pt            # Most recent (epoch 1)
│   ├── latest_model_2.pt            # Most recent (epoch 2)
│   ├── epoch_model_5.pt             # Periodic save
│   ├── epoch_model_10.pt
│   ├── ...
│   ├── epoch_model_100.pt
│   └── loss_model_best.pt           # Best by training loss
├── logs/
│   ├── train.py                     # Code snapshot
│   ├── model/                       # Model code snapshot
│   ├── log.txt                      # Text logs
│   └── events.out.tfevents.*        # TensorBoard logs
└── output/
    ├── output_TIMESTAMP_metrics.json  # Test metrics
    └── output_TIMESTAMP.pkl            # Detailed results
```

---

# Configuration Tips

## For Faster Training
```bash
--nsample 25         # Reduce sample points (4x faster)
--batch_size 40      # Increase batch size (if memory allows)
--save_interval 10   # Save less frequently
```

## For Better Accuracy
```bash
--learning_rate 1e-4    # Higher LR (if stable)
--nsample 100           # More sample points (default)
--loss_refine 1         # Apply geometric losses to refined boxes
```

## For Debugging
```bash
--debug 1           # Use validation set, only 6 epochs
--gene_layout 0     # Disable layout generation
--box_refine 0      # Disable box refinement
```

---

# Quick Reference: Important Line Numbers

| Feature | Line(s) | Description |
|---------|---------|-------------|
| LR warmup | 299-304 | 3-epoch warmup schedule |
| Refine activation | 332 | `epoch>2` check for two-stage |
| Loss weight schedule | 355 | 31-epoch gradual weights |
| Gene_ce loss | 361-385 | Gradual weighting with NaN checks |
| Box_ref_mse loss | 422-429 | Starts epoch 3, gradual weighting |
| Geometric loss refine | 391, 400, 409, 418 | Add refinement after epoch 2 |
| Validation callback | 632-669 | CURRENTLY DISABLED! |
| Gradient clipping | 461 | Clip to max_norm=1.0 |
| NaN detection | 282-294 | Helper function for debugging |
| Checkpoint saving | 708-720 | Three types of checkpoints |

---

# Next Steps

1. **Fix validation data**: Regenerate `data_valid.mat` with correct vocabulary
2. **Monitor training**: Use TensorBoard to track losses and gradients
3. **Experiment with hyperparameters**: Try different learning rates, schedules
4. **Analyze failure cases**: Use detailed test outputs to find weaknesses
5. **Improve stability**: Consider alternative training strategies

---

*End of train.py explanation*
