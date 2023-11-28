# Whisper-GUI
A simple GUI made with `gradio` to use WhisperX on Windows.  
![whisper-gui-img](images/whisper-gui.png)

## Set up
### Create a conda environment with Python 3.10
`conda create --name whisperx python=3.10`  
`conda activate whisperx`
### Install PyTorch 2.0
If you have GPU: `conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia`  
If not, for CPU: `conda install pytorch==2.0.0 torchaudio==2.0.0 cpuonly -c pytorch`
### Install whisperx
`pip install git+https://github.com/m-bain/whisperx.git`  
### Install necessary libraries
`pip install gradio==3.23.0`  
Original instructions in: https://github.com/m-bain/whisperX
### Clone this repository
`git clone https://github.com/Pikurrot/whisper-gui`
## Run the GUI
Once set up, you can just run `whisper-gui.bat` and a terminal will open, with the GUI in a new browser tab.  
Your transcriptions will be saved by default in the `outputs` folder of the repo.
