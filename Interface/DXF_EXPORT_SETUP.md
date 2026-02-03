# DXF Export Setup Guide

## Overview

The Graph2Plan interface now supports **parallel DXF export** alongside the existing .mat file format. DXF (Drawing Exchange Format) files can be opened in AutoCAD, Revit, SketchUp, and other CAD software.

## Features

✓ Exports floor plans as DXF files automatically when saving
✓ Includes exterior walls, room boundaries, windows, and doors
✓ Organized layers for easy editing in CAD software
✓ Room labels with proper positioning
✓ Configurable scale and wall thickness
✓ Non-blocking: DXF export failure won't break the save process

## Installation

### 1. Install ezdxf library

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
pip install ezdxf
```

Or if using conda:

```bash
conda install -c conda-forge ezdxf
```

### 2. Verify installation

```bash
python -c "import ezdxf; print(f'ezdxf {ezdxf.__version__} installed successfully!')"
```

### 3. Test DXF export module

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
python Houseweb/dxf_export.py
```

## Configuration

Edit `/Interface/Houseweb/views.py` to configure DXF export:

```python
# DXF Export Configuration (around line 55)
ENABLE_DXF_EXPORT = True          # Set to False to disable
DXF_SCALE = 1.0                   # Scale factor
DXF_WALL_THICKNESS = 3.0          # Wall thickness
```

### Scale Factor Options

| Scale | Description | Use Case |
|-------|-------------|----------|
| 1.0 | 1 pixel = 1 unit | Default, matches .mat files |
| 0.01 | 1 pixel = 1 cm | Metric CAD (centimeters) |
| 0.001 | 1 pixel = 1 mm | Metric CAD (millimeters) |
| 0.0254 | 1 pixel = 1 inch | Imperial CAD (inches) |
| 0.3048 | 1 pixel = 1 foot | Imperial CAD (feet) |

## How It Works

DXF export is integrated into two save endpoints:

### 1. `Save_Editbox` (views.py:985)
Called when user manually edits and saves a floor plan.

**Saves:**
- `./static/{userRoomID}.mat` (original format)
- `./static/{userRoomID}.dxf` (new parallel export)

### 2. `AdjustGraph` (views.py:689)
Called when generating layouts from the model.

**Saves:**
- `./static/{testname}.mat`
- `./static/{testname}.dxf`

## DXF File Structure

The exported DXF files contain the following layers:

| Layer | Color | Contents |
|-------|-------|----------|
| EXTERIOR_WALL | White/Gray (7) | Building outline |
| INTERIOR_WALL | Dark Gray (8) | Room partitions |
| ROOMS | Green (3) | Room boundaries (if rBoundary available) |
| WINDOWS | Cyan (4) | Window openings |
| DOORS | Magenta (6) | Door openings with swing arcs |
| LABELS | Yellow (2) | Room type labels |
| DIMENSIONS | Red (1) | Optional dimension lines |

## Usage Examples

### Basic Usage (Automatic)

Once installed, DXF files are automatically generated when you save floor plans through the web interface. Look for files like:

```
Interface/static/14926.dxf
Interface/static/14926.mat
```

### Manual Export from Python

```python
from Houseweb.dxf_export import save_floorplan_dxf
import scipy.io as sio

# Load existing .mat file
data = sio.loadmat("./static/14926.mat")
fp_data = data['data'][0, 0]

# Export to DXF with custom settings
save_floorplan_dxf(
    fp_data,
    "./static/14926_custom.dxf",
    scale=0.01,              # 1 pixel = 1 cm
    wall_thickness=5.0,      # 5cm walls
    include_labels=True,     # Add room labels
    include_dimensions=True  # Add dimension text
)
```

### Batch Conversion

Convert all existing .mat files to DXF:

```python
from Houseweb.dxf_export import batch_export_dxf

batch_export_dxf(
    mat_directory="./static",
    output_directory="./static/dxf_export",
    scale=0.01  # 1 pixel = 1 cm
)
```

## Troubleshooting

### "ezdxf not installed"

Install ezdxf:
```bash
pip install ezdxf
```

Restart the Django server after installation.

### DXF export disabled but .mat works

Check the console output:
- If you see `[Init] DXF export module loaded successfully ✓`, it's working
- If you see `[Init] DXF export not available`, ezdxf is not installed

### DXF file has wrong scale

The default scale is 1.0 (pixels). For real-world units:
- Metric: Set `DXF_SCALE = 0.01` (cm) or `0.001` (mm)
- Imperial: Set `DXF_SCALE = 0.0254` (inches)

### Disable DXF export temporarily

Set in views.py:
```python
ENABLE_DXF_EXPORT = False
```

## Opening DXF Files

### AutoCAD / AutoCAD LT
1. File → Open → Select .dxf file
2. Use Layer Manager to control visibility

### SketchUp
1. File → Import → Select .dxf file
2. Check "Merge coplanar faces"

### LibreCAD (Free)
1. File → Open → Select .dxf file

### Online Viewers
- https://sharecad.org/
- https://app.onshape.com/ (import DXF)

## Performance Notes

- DXF export adds ~50-200ms per save operation
- Export runs in parallel (non-blocking)
- Failures don't affect .mat file saving
- No impact on model inference or refinement

## File Comparison

| Feature | .MAT | .DXF |
|---------|------|------|
| Python-readable | ✓ (scipy) | ✓ (ezdxf) |
| MATLAB-readable | ✓ | ✓ |
| CAD software | ✗ | ✓ |
| Human-readable | ✗ | Partial |
| Contains raw data | ✓ | ✗ |
| Editable in CAD | ✗ | ✓ |
| File size | Smaller | Larger |

## Further Development

Potential enhancements:

1. **3D export**: Add wall heights for 3D CAD
2. **Furniture**: Export furniture as blocks
3. **Materials**: Add material properties to layers
4. **Dimensions**: Auto-generate dimension lines
5. **Export API**: Add REST endpoint for on-demand export
6. **IFC format**: Export to BIM format (Industry Foundation Classes)

## Support

For issues or questions:
- Check console output for error messages
- Verify ezdxf installation: `pip list | grep ezdxf`
- Test with: `python Houseweb/dxf_export.py`
