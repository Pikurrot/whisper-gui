@echo off

:: Activate the conda environment
call conda activate whisperx

:: Run the main Python script
python main.py --autolaunch

:: Pause (optional, only if you want to keep the console window open)
pause
