@echo off
setlocal enabledelayedexpansion

:: Variables
set CONFIG_FILE=%~dp0configs\config.json
set ENV_NAME=
set GPU_SUPPORT=
set AUTO_UPDATE=
set /a ENV_COUNT=0
set TEMP=
set PYTHON_VERSION=
set DEP_FILE=
set LOCAL_COMMIT=
set REMOTE_COMMIT=
:: Initialize list of conda environments
set MAX_ENVS=20
for /L %%x in (1,1,%MAX_ENVS%) do set "env_%%x="

:: Check conda is installed
echo Checking conda installation...
call conda --version >nul 2>&1
if %errorlevel% GEQ 1 (
	echo conda is not installed. Install Anaconda first and add conda command to PATH.
	goto cleanup
)
echo conda is installed.

:: Check if config file exists
echo Checking config file...
if exist %CONFIG_FILE% (
	goto read_env_name
)
echo Config file does not exist. Creating config file...

:: Create empty config file
type nul > %CONFIG_FILE%
echo Config file created.
goto list_envs

:: List conda environments
:list_envs
echo Listing conda environments...
for /f "skip=3 tokens=1" %%a in ('conda env list') do (
	set "env_!ENV_COUNT!=%%a"
	echo !ENV_COUNT!. %%a
	set /a ENV_COUNT+=1
)

if !ENV_COUNT! EQU 0 (
	echo No conda environments found.
	set /p TEMP= "Do you want to create a new environment named "whisperx"? ([y]/n): "
	if "!TEMP!" == "n" (
		echo Exiting...
		goto cleanup
	)
	set ENV_NAME=whisperx
	goto create_env
)

echo !ENV_COUNT!. Create a new environment (RECOMMENDED)
echo.
goto select_env

:select_env
set /p TEMP= "Select the number of the environment to use: "

:: Handle user input
if !TEMP! LSS 0 (
	echo Invalid input.
	goto select_env
)
if !TEMP! GTR !ENV_COUNT! (
	echo Invalid input.
	goto select_env
)
if "!TEMP!" == "!ENV_COUNT!" (
	set /p ENV_NAME= "Enter the name of the new environment: "
	goto create_env
)

set "ENV_NAME=!env_%TEMP%!"
echo Selected environment: "!ENV_NAME!"

:: Check python version
echo Checking python version...
call conda activate !ENV_NAME! >nul 2>&1
call python --version >nul 2>&1
if %errorlevel% GEQ 1 (
	echo Python is not installed in this environment. Install python first.
	call conda deactivate >nul 2>&1
	goto cleanup
)
for /f "tokens=2" %%a in ('python --version') do set "PYTHON_VERSION=%%a"
for /f "tokens=2 delims=." %%a in ("!PYTHON_VERSION!") do set "PYTHON_VERSION=%%a"
if !PYTHON_VERSION! LSS 10 (
	echo Incorrect python version. Install python 3.10 or newer.
	call conda deactivate >nul 2>&1
	goto cleanup
)
echo Python version is correct: !PYTHON_VERSION!
goto save_env_name

:: Create new conda environment
:create_env
echo Creating new environment "!ENV_NAME!" with python 3.10.13...
call conda create -n !ENV_NAME! python=3.10.13 -y >nul 2>&1
if %errorlevel% GEQ 1 (
	echo Failed to create environment.
	goto cleanup
)
echo Environment created.
goto activate_env

:: Read environment name from config file
:read_env_name
for /f "delims=" %%a in ('call scripts\config_read.bat "env_name"') do set ENV_NAME=%%a
set TEMP=%errorlevel%
if %TEMP% EQU 1 (
	echo Failed to read config file.
	goto cleanup
)
if %TEMP% EQU 2 (
	echo Config file does not contain "env_name" key.
	goto list_envs
)
if %TEMP% EQU 3 (
	echo "env_name" key is null in config file.
	goto list_envs
)
set TEMP="skip"
goto activate_env

:: Activate conda environment
:activate_env
echo Activating environment !ENV_NAME!...
call conda activate !ENV_NAME! >nul 2>&1
if %errorlevel% GEQ 1 (
	echo Failed to activate environment.
	goto cleanup
)
echo Environment activated.
if !TEMP! == "skip" (
	goto check_updates
)
goto save_env_name

:: save environment name to config file
:save_env_name
echo Saving environment name to config file...
call python scripts\config_write.py "env_name" "!ENV_NAME!" >nul 2>&1
if %errorlevel% GEQ 1 (
	echo Failed to save environment name to config file.
	goto cleanup
)
goto check_gpu

:: Check if GPU support is enabled
:check_gpu
echo Checking GPU support...
for /f "delims=" %%a in ('call scripts\config_read.bat "gpu_support"') do (
	set GPU_SUPPORT=%%a
	set TEMP=!errorlevel!
)
if %TEMP% EQU 0 (
	goto install_deps
)
if %TEMP% EQU 1 (
	echo Failed to read config file.
	goto cleanup
)
if %TEMP% EQU 2 (
	echo Config file does not contain "gpu_support" key.
)
if %TEMP% EQU 3 (
	echo "gpu_support" key is null in config file.
)
goto test_gpu

:test_gpu
echo Testing if GPU is available...
call nvidia-smi >nul 2>&1
if %errorlevel% EQU 0 (
	echo GPU detected.
	set /p TEMP= "Do you want to use GPU support? ([y]/n): "
	if "!TEMP!" == "y" (
		set GPU_SUPPORT=true
		echo GPU support enabled.
	) else (
		set GPU_SUPPORT=false
		echo Proceeding with CPU support.
	)
) else (
	echo No GPU detected. Proceeding with CPU support.
	set GPU_SUPPORT=false
)
echo Saving result to config file...
call python scripts\config_write.py "gpu_support" "!GPU_SUPPORT!" >nul 2>&1
if %errorlevel% GEQ 1 (
	echo Failed to save result to config file.
	goto cleanup
)
goto install_deps

:: Install corresponding dependencies
:install_deps
if "!GPU_SUPPORT!" == "true" (
	echo Installing dependencies for GPU...
	set DEP_FILE=configs\environment_gpu.yml
	echo Installing PyTorch with CUDA 11.8...
	call pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
	echo Installing dependencies for CPU...
	set DEP_FILE=configs\environment_cpu.yml
	echo Installing PyTorch...
	call pip install torch torchaudio
)
echo Istalling whisperx...
call pip install git+https://github.com/m-bain/whisperx.git
echo Installing other dependencies...
call conda env update --name !ENV_NAME! --file !DEP_FILE!
if %errorlevel% GEQ 1 (
	echo Failed to install dependencies.
	goto cleanup
)
echo Dependencies installed.
if defined LOCAL_COMMIT (
	goto set_auto_update
)
goto check_updates

:: Check for available updates
:check_updates
echo Checking for updates...
cd /d %~dp0
call git --version >nul 2>&1
if %errorlevel% GEQ 1 (
	echo git is not installed. Install git first and add git command to PATH.
	goto run
)
call git fetch >nul 2>&1
if %errorlevel% GEQ 1 (
	echo Failed to fetch updates. Check your internet connection.
	goto run
)
for /f "tokens=*" %%a in ('git rev-parse master') do set LOCAL_COMMIT=%%a
if %errorlevel% GEQ 1 (
	echo Failed to get local commit hash.
	goto run
)
if not defined LOCAL_COMMIT (
	echo Failed to get local commit hash.
	goto run
)
for /f "tokens=*" %%a in ('git rev-parse origin/master') do set REMOTE_COMMIT=%%a
if %errorlevel% GEQ 1 (
	echo Failed to get remote commit hash.
	goto run
)
if not defined REMOTE_COMMIT (
	echo Failed to get remote commit hash.
	goto run
)

for /f "delims=" %%a in ('call scripts\config_read.bat "auto_update"') do set AUTO_UPDATE=%%a
if %errorlevel% EQU 1 (
	echo Failed to read config file.
	goto run
)

if "!LOCAL_COMMIT!" == "!REMOTE_COMMIT!" (
	echo Your repository is up to date.
	goto set_auto_update
) else (
	echo Updates available.
	if "!AUTO_UPDATE!" == "true" (
		goto update_repo
	)
	set /p TEMP= "Do you want to update the repository? ([y]/n): "
	if "!TEMP!" == "n" (
		goto set_auto_update
	)
)
goto update_repo

:update_repo
echo Updating repository...
call git pull origin master >nul 2>&1
if %errorlevel% GEQ 1 (
	echo Failed to update repository.
	goto set_auto_update
)
if defined GPU_SUPPORT (
	goto install_deps
) else (
	goto check_gpu
)
goto set_auto_update

:set_auto_update
echo Checking auto update setting...
if not defined AUTO_UPDATE (
	set /p TEMP= "Do you want this repository to update automatically from now on? ([y]/n): "
	if "!TEMP!" == "n" (
		set AUTO_UPDATE=false
		echo You will be asked to update this repository every time an update is available.
		rem TODO: You can change this setting in the configuration later.
	) else (
		set AUTO_UPDATE=true
		echo This repository will update automatically from now on.
	)
	echo Saving result to config file...
	call python scripts\config_write.py "auto_update" "!AUTO_UPDATE!" >nul 2>&1
	if !errorlevel! GEQ 1 (
		echo Failed to save result to config file.
		goto run
	)
)
goto run

:run
echo Running Whisper-GUI...
call python main.py --autolaunch
goto cleanup

:cleanup
endlocal
pause
exit /b 0
