@echo off

:: Activate the conda environment
echo Activating WhisperX conda environment...
call conda activate whisperx

:: Run the main Python script
echo Running WhisperX...
python main.py --autolaunch

:: Pause (optional, only if you want to keep the console window open)
pause
