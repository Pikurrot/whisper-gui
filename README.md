# Whisper-GUI
A simple GUI made with `gradio` to use WhisperX on Windows.  

![whisper-gui-img](https://github.com/Pikurrot/Pikurrot/blob/main/images/whisper-gui/interface_screenshot.png?raw=true)

## Requirements
- [Anaconda](https://docs.anaconda.com/free/anaconda/install/windows/) installed and `conda` added to PATH.
- `git` installed and added to PATH.
- `ffmpeg` installed and added to PATH. See instructions for [Windows](https://phoenixnap.com/kb/ffmpeg-windows) or [Linux](https://phoenixnap.com/kb/install-ffmpeg-ubuntu)

## Set up
- Install
- Run the `whisper-gui.bat` file and follow the instructions. After the process, it will run the GUI in a new browser tab.

Otherwise, manual steps are:
- **Create a conda environment with Python 3.10**  
	`conda create --name whisperx python=3.10`  
	`conda activate whisperx`
- **Install PyTorch 2.0**  
	If you have GPU:  
	`conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia`  
	If not, for CPU:  
	`conda install pytorch==2.0.0 torchaudio==2.0.0 cpuonly -c pytorch`
- **Install whisperx**  
	`pip install git+https://github.com/m-bain/whisperx.git`  
	Original instructions in: https://github.com/m-bain/whisperX
- **Install necessary libraries**  
	`pip install gradio`  
- **Clone this repository**  
	`git clone https://github.com/Pikurrot/whisper-gui`

## Run the GUI
To run the program every time, you can just run the same `whisper-gui.bat`, which will also automatically check for updates of this repo.  
Your transcriptions will be saved by default in the `outputs` folder of the repo.

Otherwise, to run manually:  
`conda activate whisperx`  
`python main.py --autolaunch`

## Licensing
This project is primarily distributed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

**Third-Party Code**  
Portions of this project incorporate code from [WhisperX](https://github.com/m-bain/whisperX), which is licensed under BSD-4-Clause license. This code is used in accordance with its license, and the full text of the license can be found within the relevant [source files](scripts/whisper_model.py).
