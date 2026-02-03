# DXF Export - Quick Start Guide

## Installation (30 seconds)

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
pip install ezdxf
```

That's it! DXF export is now enabled automatically.

## What Changed?

**Before:** Only `.mat` files were saved
```
static/
  └── 14926.mat
```

**After:** Both `.mat` and `.dxf` files are saved in parallel
```
static/
  ├── 14926.mat  (original format, unchanged)
  └── 14926.dxf  (new CAD format)
```

## Testing

### 1. Test the module
```bash
python Houseweb/dxf_export.py
```

Expected output:
```
DXF Export Module
ezdxf available: True
ezdxf version: 1.x.x
```

### 2. Convert existing files
```bash
# Convert all .mat files in static/
python convert_to_dxf.py

# Convert single file
python convert_to_dxf.py --file static/14926.mat

# Convert with metric scale (1 pixel = 1 cm)
python convert_to_dxf.py --scale 0.01
```

### 3. Check the output
```bash
ls -lh static/*.dxf
```

## Configuration

Edit `Houseweb/views.py` (around line 55):

```python
ENABLE_DXF_EXPORT = True     # Turn on/off
DXF_SCALE = 1.0              # 1.0=pixels, 0.01=cm, 0.001=mm
DXF_WALL_THICKNESS = 3.0     # Wall thickness
```

## What's in the DXF File?

- ✓ Exterior boundary (building outline)
- ✓ Room boundaries (walls)
- ✓ Windows
- ✓ Doors with swing arcs
- ✓ Room labels (Living Room, Kitchen, etc.)
- ✓ Organized layers for easy CAD editing

## Opening DXF Files

**AutoCAD / AutoCAD LT:**
- File → Open → Select .dxf file

**SketchUp:**
- File → Import → Select .dxf file

**Free Options:**
- LibreCAD (https://librecad.org/)
- Online: https://sharecad.org/

## Disable DXF Export

If you want to temporarily disable it:

```python
# In views.py
ENABLE_DXF_EXPORT = False
```

## Performance Impact

- Adds ~50-200ms per save operation
- Non-blocking (doesn't slow down the UI)
- Failures don't affect .mat file saving

## Need Help?

See full documentation: `DXF_EXPORT_SETUP.md`

```bash
cat DXF_EXPORT_SETUP.md
```
