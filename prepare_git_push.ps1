# Script to prepare Git repository for pushing code changes only (no data files)

Write-Host "=== Preparing Git Repository for Push ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Remove Git LFS if installed
Write-Host "Step 1: Uninstalling Git LFS tracking..." -ForegroundColor Yellow
git lfs uninstall 2>$null
Write-Host "✓ Git LFS uninstalled" -ForegroundColor Green
Write-Host ""

# Step 2: Reset any staged changes
Write-Host "Step 2: Resetting staged changes..." -ForegroundColor Yellow
git reset HEAD .
Write-Host "✓ Staging area cleared" -ForegroundColor Green
Write-Host ""

# Step 3: Remove data files from Git tracking (keeps them on disk)
Write-Host "Step 3: Removing large data files from Git tracking..." -ForegroundColor Yellow
git rm -r --cached DataPreparation/data/ 2>$null
git rm -r --cached Network/data/ 2>$null
git rm -r --cached Interface/retrieval/ 2>$null
git rm -r --cached Interface/static/Data/ 2>$null
git rm -r --cached experiment/ 2>$null
git rm --cached .gitattributes 2>$null
Write-Host "✓ Data files removed from Git tracking" -ForegroundColor Green
Write-Host ""

# Step 4: Add only code changes
Write-Host "Step 4: Adding code changes..." -ForegroundColor Yellow
git add .gitignore
git add Network/train.py
git add DataPreparation/6.cluster.py
git add DataPreparation/check_ids.py
git add reduce_dataset.py
git add reduce_dataset_simple.py
git add CPU_TESTING_GUIDE.md
git add requirements.txt
Write-Host "✓ Code changes staged" -ForegroundColor Green
Write-Host ""

# Step 5: Show status
Write-Host "Step 5: Current Git status:" -ForegroundColor Yellow
git status
Write-Host ""

# Step 6: Instructions
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To commit and push your changes, run:" -ForegroundColor White
Write-Host '  git commit -m "Training improvements: CPU compatibility, validation fixes, dataset reduction"' -ForegroundColor Green
Write-Host "  git push origin ai-training-branch" -ForegroundColor Green
Write-Host ""
Write-Host "Your data files are now ignored and will stay on your local machine only." -ForegroundColor Yellow
