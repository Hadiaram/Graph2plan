# DXF Export Button - Setup Complete! 📐

## What Was Added

A new **"📐 Export DXF"** button has been added to your interface that saves floor plans to your network location:
```
N:\IT-TEST\Hadi Aram Bashir\DXF Floor Plans
```

## Button Location

The DXF Export button appears next to the Refine button:

```
[Save] [🔧 Refine] [📐 Export DXF] [Transfer] [Delete Mode: OFF] [Reset]
```

- **Color:** Blue (#2196F3)
- **Position:** After the Refine button
- **Visibility:** Shows only after a floor plan is generated

## Installation Steps

### 1. Install ezdxf library

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
pip install ezdxf
```

### 2. Verify network drive is accessible

From WSL, check if you can access the N: drive:

```bash
# Test if path exists
ls /mnt/n/IT-TEST/Hadi\ Aram\ Bashir/

# If the above fails, the N: drive might not be mounted in WSL
# You can mount it using Windows drive mapping
```

**If N: drive is not accessible from WSL:**

The path will automatically work when running Django on Windows (not in WSL). Alternatively, you can update the path in `views.py` line 61 to use a different accessible location.

### 3. Restart Django server

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
python manage.py runserver
```

## How to Use

1. **Generate a floor plan** using the interface (or load an existing one)
2. **Save the floor plan** using the green "Save" button
3. **Click "📐 Export DXF"** button
4. Wait for confirmation message with export details
5. **Check the network location** for your DXF file

## Success Message

When export succeeds, you'll see:

```
✅ DXF Export Complete!

File: 14926.dxf
Location: N:\IT-TEST\Hadi Aram Bashir\DXF Floor Plans\
Size: 45.2 KB
Rooms: 8
Scale: 1.0 (1 pixel = 1.0 units)
```

## Error Messages

### "Please save a floor plan first!"
**Cause:** No floor plan loaded
**Fix:** Generate or load a floor plan, then click Save before exporting

### "Floor plan 14926.mat not found"
**Cause:** The .mat file doesn't exist in static/ directory
**Fix:** Click the "Save" button first to save the floor plan

### "Cannot access network path"
**Cause:** N: drive not accessible
**Fix:**
1. Check if N: drive is mapped in Windows File Explorer
2. If using WSL, the drive might not be available
3. Update the path in `views.py` to a local path for testing:
   ```python
   DXF_SAVE_PATH = "./static/dxf_output"  # Temporary local path
   ```

### "DXF export not available"
**Cause:** ezdxf library not installed
**Fix:** Run `pip install ezdxf` and restart Django server

## Configuration

Edit `/Interface/Houseweb/views.py` (around line 58-64):

```python
# DXF Export Configuration
ENABLE_AUTO_DXF_EXPORT = False  # Keep disabled (button only)
DXF_SCALE = 1.0                 # Scale: 1.0=pixels, 0.01=cm, 0.001=mm
DXF_WALL_THICKNESS = 3.0        # Wall thickness in drawing units

# DXF Save Path - Network location
DXF_SAVE_PATH = r"/mnt/n/IT-TEST/Hadi Aram Bashir/DXF Floor Plans"
```

### Change Scale

To export with different units, change `DXF_SCALE`:

| Scale | Meaning | Example Use |
|-------|---------|-------------|
| 1.0 | 1 pixel = 1 unit | Default, matches visual |
| 0.01 | 1 pixel = 1 cm | Metric drawings (cm) |
| 0.001 | 1 pixel = 1 mm | Metric drawings (mm) |
| 0.0254 | 1 pixel = 1 inch | Imperial drawings |

### Change Save Location

Update `DXF_SAVE_PATH` to any accessible path:

```python
# Local folder (for testing)
DXF_SAVE_PATH = "./static/dxf_output"

# Different network drive
DXF_SAVE_PATH = r"/mnt/z/MyProject/FloorPlans"

# Windows path (if running on Windows, not WSL)
DXF_SAVE_PATH = r"N:\IT-TEST\Hadi Aram Bashir\DXF Floor Plans"
```

## Files Modified

1. **`Houseweb/views.py`**
   - Added `Export_DXF()` endpoint (line 1792)
   - Added configuration variables (line 58-64)
   - Disabled auto-export (now manual only)

2. **`templates/home.html`**
   - Added "📐 Export DXF" button (line 855)

3. **`static/js/buttonEvent.js`**
   - Added button show logic (line 1139)
   - Added click handler with AJAX call (line 1461)

4. **`House/urls.py`**
   - Added URL mapping for Export_DXF (line 42)

## Testing

### Test Network Path Accessibility

```bash
# From WSL
touch "/mnt/n/IT-TEST/Hadi Aram Bashir/DXF Floor Plans/test.txt"

# If this works, DXF export should work
# If it fails, check Windows drive mapping
```

### Test DXF Module

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
python -c "from Houseweb.dxf_export import HAS_EZDXF; print('DXF Export:', 'ENABLED' if HAS_EZDXF else 'DISABLED')"
```

Expected output:
```
[Init] DXF export module loaded successfully ✓
DXF Export: ENABLED
```

### Test Export Manually

If the button doesn't work, you can test the export function directly in Django shell:

```bash
python manage.py shell
```

```python
from Houseweb.dxf_export import save_floorplan_dxf
import scipy.io as sio
import os

# Load a .mat file
data = sio.loadmat("./static/14926.mat")
fp_data = data['data'][0, 0]

# Try local export first (to test ezdxf works)
success = save_floorplan_dxf(fp_data, "./static/test.dxf")
print(f"Local export: {'SUCCESS' if success else 'FAILED'}")

# Try network export (to test path access)
success = save_floorplan_dxf(fp_data, "/mnt/n/IT-TEST/Hadi Aram Bashir/DXF Floor Plans/test.dxf")
print(f"Network export: {'SUCCESS' if success else 'FAILED'}")
```

## Automatic vs Manual Export

**Current Setup:** Manual only (button-triggered)

- ✅ **Manual Export:** Enabled via button
- ❌ **Automatic Export:** Disabled (doesn't save DXF on every save)

To enable automatic export (saves DXF every time you save .mat):

```python
# In views.py line 58
ENABLE_AUTO_DXF_EXPORT = True  # Change to True
```

Not recommended - better to keep it manual so you control when DXF files are created.

## Troubleshooting Checklist

- [ ] ezdxf installed: `pip list | grep ezdxf`
- [ ] Django server restarted after installation
- [ ] N: drive mapped in Windows
- [ ] Network path accessible: `ls /mnt/n/IT-TEST/`
- [ ] Floor plan saved before clicking Export DXF
- [ ] Browser console shows no JavaScript errors (F12)
- [ ] Django console shows DXF module loaded message

## Opening Exported DXF Files

The exported DXF files can be opened in:

- **AutoCAD / AutoCAD LT** (Professional CAD)
- **SketchUp** (3D modeling)
- **LibreCAD** (Free, open-source)
- **Revit** (BIM software)
- **Online viewers:** https://sharecad.org/

## Next Steps

1. Install ezdxf
2. Verify network path access
3. Restart Django server
4. Test the button with a saved floor plan
5. Check `N:\IT-TEST\Hadi Aram Bashir\DXF Floor Plans\` for the file

## Need Help?

**Common Issues:**

1. **Button doesn't appear:** Make sure you've generated/loaded a floor plan first
2. **Network path error:** Try changing to a local path temporarily for testing
3. **Import error:** Make sure ezdxf is installed: `pip install ezdxf`

**Console Logs:**

Check Django console for messages like:
```
[DXF Export] Loading ./static/14926.mat...
[DXF Export] Exporting to /mnt/n/IT-TEST/...
[DXF Export] ✓ Successfully exported 14926.dxf (45.2 KB)
```

Check browser console (F12) for messages like:
```
[DXF Export] Exporting floor plan: 14926
[DXF Export] Success! {filename: "14926.dxf", ...}
```
