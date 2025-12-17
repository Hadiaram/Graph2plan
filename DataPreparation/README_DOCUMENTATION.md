# Graph2plan Custom Dataset Documentation

**Navigation Guide for Custom Dataset Integration**

This document provides the reading order for all documentation files related to integrating custom datasets with Graph2plan.

---

## Documentation Reading Order

### 1. **README.md** (Original Graph2plan Documentation)
📍 **Location**: `DataPreparation/README.md`

**Purpose**: Understanding the original Graph2plan data format and pipeline

**Read this to learn:**
- Graph2plan's data structure (8 required fields)
- Room categories (18 types) and edge relations (10 types)
- Original data preparation workflow
- What each script (1-6) does

**Key Sections:**
- Data format specification
- Field descriptions (boundary, rType, rBoundary, gtBox, gtBoxNew, rEdge, order)
- Original RPLAN dataset structure

**When to read**: First - establishes foundation

---

### 2. **CUSTOM_DATASET_GUIDE.md** (General Custom Dataset Integration)
📍 **Location**: `DataPreparation/CUSTOM_DATASET_GUIDE.md`

**Purpose**: General guide for integrating any custom dataset

**Read this to learn:**
- Do you need to retrain? (Answer: NO)
- Graph2plan's two-stage architecture (retrieval + generation)
- Why no retraining is needed
- Generic workflow for custom datasets
- Three utility scripts (convert_mat_to_python, recombine_split_dataset, convert_resplan_to_mat)
- Data preparation pipeline overview
- Interface file locations and paths

**Key Sections:**
- Architecture explanation (Turn Function matching)
- Required vs optional data preparation scripts
- Workflow summary (8 main steps)
- Script reference with examples
- Troubleshooting common issues
- File locations reference

**When to read**: Second - general concepts and workflow

---

### 3. **RESPLAN_INTEGRATION_GUIDE.md** (ResPlan-Specific Integration)
📍 **Location**: `DataPreparation/RESPLAN_INTEGRATION_GUIDE.md`

**Purpose**: Detailed guide specifically for ResPlan dataset integration

**Read this to learn:**
- ResPlan dataset structure (graph + Shapely geometries)
- How ResPlan differs from RPLAN format
- Enhanced conversion script features
- Complete step-by-step integration process
- Train/test split generation
- Detailed data preparation scripts walkthrough
- ResPlan-specific troubleshooting

**Key Sections:**
- ResPlan structure vs Graph2plan format comparison
- Shapely geometry handling
- Graph node parsing and room extraction
- Boundary extraction and front door alignment
- Room type filtering (excluding wall/window/door)
- Visual process diagrams
- Complete command reference
- 8 common ResPlan issues with solutions

**When to read**: Third - if using ResPlan dataset format

---

## Quick Navigation by Task

### I want to understand Graph2plan's data format
→ **README.md** (Section 1: Data Format)

### I want to know if I need to retrain the model
→ **CUSTOM_DATASET_GUIDE.md** (Section 2: Do You Need to Retrain?)

### I want to convert a MATLAB .mat file to Python
→ **CUSTOM_DATASET_GUIDE.md** (Script: convert_mat_to_python.py)

### I want to recombine split floor plan files
→ **CUSTOM_DATASET_GUIDE.md** (Script: recombine_split_dataset.py)

### I have a ResPlan dataset with Shapely geometries
→ **RESPLAN_INTEGRATION_GUIDE.md** (Complete guide)

### I need to create train.txt and test.txt files
→ **RESPLAN_INTEGRATION_GUIDE.md** (Section: Creating Train/Test Splits)

### I'm running data preparation scripts 1-6
→ **RESPLAN_INTEGRATION_GUIDE.md** (Section: Running Data Preparation Scripts)

### I need to update the Interface with new files
→ **RESPLAN_INTEGRATION_GUIDE.md** (Section: Updating the Interface)

### I'm getting conversion errors
→ **RESPLAN_INTEGRATION_GUIDE.md** (Section: Troubleshooting ResPlan Issues)
→ **CUSTOM_DATASET_GUIDE.md** (Section: Troubleshooting)

---

## Documentation Summary

| Document | Pages | Focus | Audience |
|----------|-------|-------|----------|
| **README.md** | ~3 | Original format | All users |
| **CUSTOM_DATASET_GUIDE.md** | ~25 | General integration | Custom dataset users |
| **RESPLAN_INTEGRATION_GUIDE.md** | ~40 | ResPlan specifics | ResPlan users |

---

## Complete Workflow Reference

### For ResPlan Dataset Users (Recommended Path)

1. **Read README.md** → Understand Graph2plan format
2. **Read CUSTOM_DATASET_GUIDE.md** → Learn general workflow
3. **Read RESPLAN_INTEGRATION_GUIDE.md** → Follow specific steps
4. **Execute commands** → Run conversion and preparation
5. **Refer back as needed** → Use troubleshooting sections

### For Other Custom Dataset Users

1. **Read README.md** → Understand Graph2plan format
2. **Read CUSTOM_DATASET_GUIDE.md** → Complete guide
3. **Customize conversion script** → Adapt convert_item() function
4. **Execute commands** → Run preparation scripts
5. **Refer back as needed** → Use troubleshooting sections

---

## Additional Files

### Script Files
- `convert_resplan_to_mat.py` - ResPlan PKL → Graph2plan MAT conversion
- `convert_mat_to_python.py` - MAT → Python formats (pkl, json, csv, npz)
- `recombine_split_dataset.py` - Merge split floor plan files
- `create_train_test_split.py` - Generate train.txt and test.txt
- `1.tf_train.py` through `6.cluster.py` - Data preparation pipeline

### Configuration
- `config.py` - Points to your data.mat file

### Generated Data
- `data/` directory - All generated intermediate files

---

## Flowchart: Which Guide Should I Read?

```
Start Here
    ↓
Do you understand Graph2plan's data format?
    ├─ NO  → Read README.md first
    └─ YES → Continue
           ↓
Is your dataset in ResPlan format?
(Has 'graph' and 'inner' fields with Shapely geometries?)
    ├─ YES → Read CUSTOM_DATASET_GUIDE.md (overview)
    │         Then read RESPLAN_INTEGRATION_GUIDE.md (detailed steps)
    │         → Execute ResPlan workflow
    │
    └─ NO  → Read CUSTOM_DATASET_GUIDE.md (complete)
              → Customize convert_item() function
              → Execute generic workflow
```

---

## Key Concepts by Document

### README.md
- 8 required fields per floor plan
- 18 room categories (0-17)
- 10 edge relation types (0-9)
- Turn Function (TF) for geometric matching

### CUSTOM_DATASET_GUIDE.md
- No retraining needed (geometric retrieval + pre-trained generation)
- Two-stage architecture explained
- Generic conversion workflow
- Three utility scripts
- Interface file locations

### RESPLAN_INTEGRATION_GUIDE.md
- ResPlan structure (graph + Shapely)
- Automatic room filtering
- Graph node parsing
- Boundary extraction from 'inner'
- Front door alignment
- Train/test split generation
- Complete command reference
- Troubleshooting with solutions

---

## Common Questions

**Q: Do I need to read all three documents?**
A: Start with README.md, then CUSTOM_DATASET_GUIDE.md. Only read RESPLAN_INTEGRATION_GUIDE.md if using ResPlan format.

**Q: I just want to use ResPlan dataset. What's the minimum?**
A: Read README.md (10 min), skim CUSTOM_DATASET_GUIDE.md (15 min), follow RESPLAN_INTEGRATION_GUIDE.md Step-by-Step section (execute commands).

**Q: Where are the troubleshooting sections?**
A:
- Generic issues: CUSTOM_DATASET_GUIDE.md → Troubleshooting
- ResPlan issues: RESPLAN_INTEGRATION_GUIDE.md → Troubleshooting ResPlan Issues

**Q: I'm getting an error. Which guide has the solution?**
A:
1. Check error message
2. If during conversion: RESPLAN_INTEGRATION_GUIDE.md → Troubleshooting
3. If during data preparation: CUSTOM_DATASET_GUIDE.md → Troubleshooting
4. If in Interface: CUSTOM_DATASET_GUIDE.md → File Locations Reference

**Q: Can I skip README.md?**
A: Not recommended. It explains the data format that all other guides reference.

---

## Reading Time Estimates

- **README.md**: 10 minutes
- **CUSTOM_DATASET_GUIDE.md**: 30-45 minutes (skim), 60-90 minutes (detailed)
- **RESPLAN_INTEGRATION_GUIDE.md**: 45-60 minutes (skim), 2-3 hours (detailed with examples)

**Total for ResPlan users**: ~2 hours to read + understand
**Execution time**: 1-2 hours (conversion + data prep)

---

## Document Status

| Document | Status | Last Updated | Version |
|----------|--------|--------------|---------|
| README.md | Original | - | Graph2plan v1.0 |
| CUSTOM_DATASET_GUIDE.md | Complete | 2025-12-16 | v1.0 |
| RESPLAN_INTEGRATION_GUIDE.md | Complete | 2025-12-16 | v1.0 |
| README_DOCUMENTATION.md | Complete | 2025-12-16 | v1.0 |

---

## Quick Command Reference

See each document for detailed explanations:

**Conversion (ResPlan → MAT):**
```bash
python convert_resplan_to_mat.py --convert input.pkl --output data.mat
```
📖 Details: RESPLAN_INTEGRATION_GUIDE.md → Step 1

**Train/Test Split:**
```bash
python create_train_test_split.py --train-ratio 0.85
```
📖 Details: RESPLAN_INTEGRATION_GUIDE.md → Creating Train/Test Splits

**Data Preparation (Scripts 1-6):**
```bash
python 1.tf_train.py
python 2.data_train_converted.py
python 3.rNum_train.py
python 4.data_train_eNum.py
python 6.cluster.py
```
📖 Details: RESPLAN_INTEGRATION_GUIDE.md → Running Data Preparation Scripts

---

## Support

- **Issues**: Check Troubleshooting sections in respective guides
- **Questions**: Refer to appropriate guide based on flowchart above
- **Original Paper**: SIGGRAPH 2020 - Graph2plan
- **Original Authors**: Ruizhen Hu, Zeyu Huang, Yuhan Tang

---

**Generated**: 2025-12-16
**Purpose**: Navigation guide for Graph2plan custom dataset documentation
**Maintained by**: Claude Code
