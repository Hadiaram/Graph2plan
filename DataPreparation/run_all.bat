@echo off
echo ========================================
echo Running DataPreparation Pipeline
echo ========================================
echo.

echo [Step 1/6] Running 1.tf_train.py...
python 1.tf_train.py
if %errorlevel% neq 0 (
    echo ERROR: Step 1 failed!
    pause
    exit /b %errorlevel%
)
echo Step 1 completed successfully.
echo.

echo [Step 2/6] Running 2.data_train_converted.py...
python 2.data_train_converted.py
if %errorlevel% neq 0 (
    echo ERROR: Step 2 failed!
    pause
    exit /b %errorlevel%
)
echo Step 2 completed successfully.
echo.

echo [Step 3/6] Running 3.rNum_train.py...
python 3.rNum_train.py
if %errorlevel% neq 0 (
    echo ERROR: Step 3 failed!
    pause
    exit /b %errorlevel%
)
echo Step 3 completed successfully.
echo.

echo [Step 4/6] Running 4.data_train_eNum.py...
python 4.data_train_eNum.py
if %errorlevel% neq 0 (
    echo ERROR: Step 4 failed!
    pause
    exit /b %errorlevel%
)
echo Step 4 completed successfully.
echo.

echo [Step 5/6] Running 5.data_test_converted.py...
python 5.data_test_converted.py
if %errorlevel% neq 0 (
    echo ERROR: Step 5 failed!
    pause
    exit /b %errorlevel%
)
echo Step 5 completed successfully.
echo.

echo [Step 6/6] Running 6.cluster.py...
python 6.cluster.py
if %errorlevel% neq 0 (
    echo ERROR: Step 6 failed!
    pause
    exit /b %errorlevel%
)
echo Step 6 completed successfully.
echo.

echo ========================================
echo All steps completed successfully!
echo ========================================
pause
