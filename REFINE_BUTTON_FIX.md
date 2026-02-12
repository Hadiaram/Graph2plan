# Refine Button - "Not Found" Error Fix

## Problem Confirmed

The endpoint **IS working** when tested via curl. The issue is with browser caching.

## Test Result

```bash
curl http://localhost:8000/index/Refine_Floorplan/?userRoomID=test
# Returns: {"success": false, "error": "Floor plan test not found"}
```

✅ This confirms the endpoint is responding!

## Solution Steps

### 1. **Hard Refresh the Browser**

Clear cached JavaScript:

- **Chrome/Edge**: Ctrl + Shift + R or Ctrl + F5
- **Firefox**: Ctrl + Shift + R
- **Safari**: Cmd + Shift + R

### 2. **Clear Browser Cache** (if hard refresh doesn't work)

- Open DevTools (F12)
- Go to Application/Storage tab
- Click "Clear site data" or "Clear storage"
- Reload the page

### 3. **Verify the Django Server is Running**

Make sure you restarted the server after URL changes:

```bash
cd C:\Users\hmbashir\source\Graph2plan\Interface
python manage.py runserver
```

### 4. **Test the Endpoint Manually**

Open in browser:

```text
http://localhost:8000/index/Refine_Floorplan/?userRoomID=test
```

You should see JSON response:

```json
{"success": false, "error": "Floor plan test not found"}
```

If you see this, the endpoint is working!

### 5. **Check Browser Console**

Open DevTools Console (F12) and look for:

```text
[Refine Button] Calling refinement for: <filename>
```

The filename should be in format: `something.png.mat`

## Common Issues

### Issue 1: File Path

If you get "Floor plan X not found", check:

- The file exists in `Interface/static/` directory
- The filename format is correct: `name.png.mat`
- The cookie `hsname` is set (check in DevTools > Application > Cookies)

### Issue 2: Server Not Restarted

- Django doesn't auto-reload URL configuration changes
- You MUST restart the server after editing `urls.py`

### Issue 3: Static File Caching

- Django development server caches static files
- Use Ctrl + Shift + R to force reload JavaScript

## Verification

Run this command to test the endpoint:

```bash
curl http://localhost:8000/index/Refine_Floorplan/?userRoomID=test
```

If you see JSON error about file not found, **the endpoint is working correctly!**
