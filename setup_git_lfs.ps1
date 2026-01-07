# Git LFS Setup Script for ResPlan Dataset
# Run this script from the project root directory

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Git LFS Setup for ResPlan Dataset" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git LFS is installed
Write-Host "Step 1: Checking Git LFS installation..." -ForegroundColor Yellow
try {
    $lfsVersion = git lfs version 2>&1
    Write-Host "✓ Git LFS is installed: $lfsVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git LFS is NOT installed!" -ForegroundColor Red
    Write-Host "Please install from: https://git-lfs.github.com/" -ForegroundColor Red
    Write-Host "Or run: winget install -e --id GitHub.GitLFS" -ForegroundColor Yellow
    exit 1
}

# Initialize Git LFS
Write-Host "`nStep 2: Initializing Git LFS..." -ForegroundColor Yellow
git lfs install
Write-Host "✓ Git LFS initialized" -ForegroundColor Green

# Track .mat files
Write-Host "`nStep 3: Configuring Git LFS to track .mat files..." -ForegroundColor Yellow
git lfs track "Network/data/*.mat"
Write-Host "✓ Git LFS will track: Network/data/*.mat" -ForegroundColor Green

# Display .gitattributes
Write-Host "`nStep 4: Verifying .gitattributes..." -ForegroundColor Yellow
if (Test-Path ".gitattributes") {
    Get-Content ".gitattributes"
    Write-Host "✓ .gitattributes created/updated" -ForegroundColor Green
} else {
    Write-Host "✗ .gitattributes not found!" -ForegroundColor Red
}

# Stage files
Write-Host "`nStep 5: Staging files..." -ForegroundColor Yellow
git add .gitattributes
git add .gitignore
git add Network/data/*.mat
Write-Host "✓ Files staged" -ForegroundColor Green

# Verify LFS tracking
Write-Host "`nStep 6: Verifying Git LFS is tracking files..." -ForegroundColor Yellow
$lfsFiles = git lfs ls-files
if ($lfsFiles) {
    Write-Host "✓ Git LFS is tracking:" -ForegroundColor Green
    git lfs ls-files
} else {
    Write-Host "⚠ No files tracked by LFS yet (this is OK before commit)" -ForegroundColor Yellow
}

# Show status
Write-Host "`nStep 7: Git status..." -ForegroundColor Yellow
git status --short

# Summary
Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review the staged changes above" -ForegroundColor White
Write-Host "2. Commit with:" -ForegroundColor White
Write-Host '   git commit -m "Setup Git LFS for ResPlan dataset"' -ForegroundColor Cyan
Write-Host "3. Push to remote:" -ForegroundColor White
Write-Host "   git push origin ai-training-branch" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your data files (127 MB total) will be stored in Git LFS!" -ForegroundColor Green
