@echo off
REM ============================================================
REM Data Preparation Pipeline for Graph2Plan
REM Runs scripts 1-6 in sequence to prepare training data
REM ============================================================

echo.
echo ============================================================
echo Graph2Plan Data Preparation Pipeline
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

echo Starting data preparation...
echo.

REM ============================================================
REM Step 1: Generate turning functions
REM ============================================================
echo [1/6] Running 1.tf_train.py - Generating turning functions...
python 1.tf_train.py
if errorlevel 1 (
    echo ERROR: Step 1 failed
    pause
    exit /b 1
)
echo Step 1 complete!
echo.

REM ============================================================
REM Step 2: Convert training data
REM ============================================================
echo [2/6] Running 2.data_train_converted.py - Converting training data...
python 2.data_train_converted.py
if errorlevel 1 (
    echo ERROR: Step 2 failed
    pause
    exit /b 1
)
echo Step 2 complete!
echo.

REM ============================================================
REM Step 3: Count room numbers
REM ============================================================
echo [3/6] Running 3.rNum_train.py - Counting room types...
python 3.rNum_train.py
if errorlevel 1 (
    echo ERROR: Step 3 failed
    pause
    exit /b 1
)
echo Step 3 complete!
echo.

REM ============================================================
REM Step 4: Generate edge numbers
REM ============================================================
echo [4/6] Running 4.data_train_eNum.py - Generating edge numbers...
python 4.data_train_eNum.py
if errorlevel 1 (
    echo ERROR: Step 4 failed
    pause
    exit /b 1
)
echo Step 4 complete!
echo.

REM ============================================================
REM Step 5: Convert test data
REM ============================================================
echo [5/6] Running 5.data_test_converted.py - Converting test data...
python 5.data_test_converted.py
if errorlevel 1 (
    echo ERROR: Step 5 failed
    pause
    exit /b 1
)
echo Step 5 complete!
echo.

REM ============================================================
REM Step 6: Cluster training data (requires FAISS)
REM ============================================================
echo [6/6] Running 6.cluster.py - Clustering training data...
python 6.cluster.py
if errorlevel 1 (
    echo ERROR: Step 6 failed
    echo Note: This step requires FAISS library
    echo Install with: pip install faiss-cpu
    pause
    exit /b 1
)
echo Step 6 complete!
echo.

echo ============================================================
echo Data preparation complete!
echo ============================================================
echo.
echo Generated files:
echo   - data/trainTF.pkl, data/testTF.pkl
echo   - Interface/retrieval/tf_train.npy
echo   - Interface/retrieval/D_test_train.npy
echo   - Interface/static/Data/data_train_converted.pkl
echo   - Interface/static/Data/rNum_train.npy
echo   - Interface/static/Data/data_train_eNum.pkl
echo   - Interface/static/Data/data_test_converted.pkl
echo   - Interface/retrieval/centroids_train.npy
echo   - Interface/retrieval/clusters_train.npy
echo.
echo You can now run training with:
echo   cd ..\Network
echo   python train.py --batch_size 20 --epoch 50 --learning_rate 0.0001 --workers 2
echo.
pause
