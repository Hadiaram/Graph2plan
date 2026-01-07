@echo off
REM Git LFS Setup for ALL Large Data Files
REM Total: ~876 MB across multiple directories

echo ======================================
echo Git LFS Setup for ALL Data Files
echo ======================================
echo.
echo Total data size: ~876 MB
echo.

REM Initialize Git LFS
echo Step 1: Initializing Git LFS...
git lfs install
echo [OK] Git LFS initialized
echo.

REM Track all .mat files (127 MB)
echo Step 2: Tracking .mat files...
git lfs track "*.mat"
git lfs track "**/*.mat"
echo [OK] Tracking all .mat files

REM Track all .pkl files (487 MB)
echo Step 3: Tracking .pkl files...
git lfs track "*.pkl"
git lfs track "**/*.pkl"
echo [OK] Tracking all .pkl files

REM Track all .npy files (262 MB)
echo Step 4: Tracking .npy files...
git lfs track "*.npy"
git lfs track "**/*.npy"
echo [OK] Tracking all .npy files

echo.
echo Step 5: Showing .gitattributes...
type .gitattributes
echo.

REM Stage .gitattributes
echo Step 6: Staging .gitattributes...
git add .gitattributes
echo [OK]
echo.

REM Stage all the data files
echo Step 7: Staging ALL data files...
echo This may take a moment...
git add Network/data/*.mat
git add Network/data.mat
git add DataPreparation/*.pkl
git add DataPreparation/data/*.npy
git add DataPreparation/data/*.pkl
git add Interface/retrieval/*.npy
git add Interface/static/**/*.pkl
echo [OK] All data files staged
echo.

REM Verify LFS tracking
echo Step 8: Verifying Git LFS files...
git lfs ls-files
echo.

REM Show status
echo Step 9: Git status...
git status --short
echo.

REM Summary
echo ======================================
echo Setup Complete!
echo ======================================
echo.
echo Files tracked with Git LFS:
echo - Network/data/*.mat (127 MB)
echo - DataPreparation/*.pkl (382 MB)
echo - DataPreparation/data/*.npy, *.pkl (145 MB)
echo - Interface/retrieval/*.npy (122 MB)
echo - Interface/static/**/*.pkl (101 MB)
echo.
echo Total: ~876 MB tracked with LFS
echo.
echo Next steps:
echo 1. Review the staged changes above
echo 2. Commit:
echo    git commit -m "Add all data files with Git LFS"
echo 3. Push (this will upload 876 MB to LFS):
echo    git push origin ai-training-branch
echo.
echo NOTE: This is a large upload! Make sure you have:
echo - Good internet connection
echo - GitHub LFS bandwidth (free tier = 1GB/month)
echo.
pause
