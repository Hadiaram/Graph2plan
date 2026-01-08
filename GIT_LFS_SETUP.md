# Git LFS Setup - Step by Step Guide
# Run these commands in PowerShell from your project root

# ============================================
# STEP 1: Verify Git LFS is installed
# ============================================
git lfs version
# If you see an error, install Git LFS first:
# Download from: https://git-lfs.github.com/
# Or: winget install -e --id GitHub.GitLFS

# ============================================
# STEP 2: Initialize Git LFS in your repo
# ============================================
git lfs install

# ============================================
# STEP 3: Configure Git LFS to track .mat files
# ============================================
# This tells Git LFS to track all .mat files in Network/data/
git lfs track "Network/data/*.mat"

# Verify the .gitattributes file was created
cat .gitattributes

# ============================================
# STEP 4: Stage the .gitattributes file
# ============================================
git add .gitattributes

# ============================================
# STEP 5: Stage your updated .gitignore
# ============================================
git add .gitignore

# ============================================
# STEP 6: Stage the data files (Git LFS will handle them)
# ============================================
git add Network/data/data_train.mat
git add Network/data/data_valid.mat
git add Network/data/data_test.mat

# Or add all at once:
# git add Network/data/*.mat

# ============================================
# STEP 7: Verify Git LFS is tracking the files
# ============================================
git lfs ls-files
# You should see your three .mat files listed

# ============================================
# STEP 8: Check what's staged
# ============================================
git status

# ============================================
# STEP 9: Commit the changes
# ============================================
git commit -m "Setup Git LFS for ResPlan dataset

- Configure Git LFS to track Network/data/*.mat files
- Update .gitignore to allow Git LFS tracking
- Add ResPlan dataset files (127 MB total):
  * data_train.mat (51.26 MB)
  * data_valid.mat (19.13 MB)
  * data_test.mat (56.45 MB)"

# ============================================
# STEP 10: Push to remote
# ============================================
git push origin ai-training-branch

# Git LFS will automatically upload the large files to LFS storage
# and commit pointer files to the regular Git repository

# ============================================
# VERIFICATION
# ============================================
# After pushing, verify on GitHub that the files show the LFS badge
# The files should display "Stored with Git LFS" on GitHub

# ============================================
# TROUBLESHOOTING
# ============================================

# If files were already committed before LFS:
# You need to remove them from Git history first:
# git rm --cached Network/data/*.mat
# git commit -m "Remove .mat files from Git (preparing for LFS)"
# Then follow steps 3-10 above

# If you get "pointer file" errors:
# git lfs migrate import --include="Network/data/*.mat"

# Check LFS status:
# git lfs status

# See LFS file sizes:
# git lfs ls-files -s
