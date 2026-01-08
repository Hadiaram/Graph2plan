@echo off
REM Git LFS Setup Script for ResPlan Dataset
REM Run this from the project root directory

echo ======================================
echo Git LFS Setup for ResPlan Dataset
echo ======================================
echo.

REM Check if Git LFS is installed
echo Step 1: Checking Git LFS installation...
git lfs version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git LFS is NOT installed!
    echo Please install from: https://git-lfs.github.com/
    echo Or run: winget install -e --id GitHub.GitLFS
    pause
    exit /b 1
)
echo [OK] Git LFS is installed
echo.

REM Initialize Git LFS
echo Step 2: Initializing Git LFS...
git lfs install
echo [OK] Git LFS initialized
echo.

REM Track .mat files
echo Step 3: Configuring Git LFS to track .mat files...
git lfs track "Network/data/*.mat"
echo [OK] Git LFS will track: Network/data/*.mat
echo.

REM Display .gitattributes
echo Step 4: Verifying .gitattributes...
if exist .gitattributes (
    type .gitattributes
    echo [OK] .gitattributes created/updated
) else (
    echo [ERROR] .gitattributes not found!
)
echo.

REM Stage files
echo Step 5: Staging files...
git add .gitattributes
git add .gitignore
git add Network/data/*.mat
echo [OK] Files staged
echo.

REM Verify LFS tracking
echo Step 6: Verifying Git LFS tracking...
git lfs ls-files
echo.

REM Show status
echo Step 7: Git status...
git status --short
echo.

REM Summary
echo ======================================
echo Setup Complete!
echo ======================================
echo.
echo Next steps:
echo 1. Review the staged changes above
echo 2. Commit with:
echo    git commit -m "Setup Git LFS for ResPlan dataset"
echo 3. Push to remote:
echo    git push origin ai-training-branch
echo.
echo Your data files (127 MB total) will be stored in Git LFS!
echo.
pause
