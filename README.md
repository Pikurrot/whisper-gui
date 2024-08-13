# Whisper-GUI
A simple GUI made with `gradio` to use Whisper.  

![whisper-gui-img](https://github.com/Pikurrot/Pikurrot/blob/main/images/whisper-gui/interface_screenshot.png?raw=true)

## Requirements
- [Anaconda](https://docs.anaconda.com/free/anaconda/install/) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) installed and `conda` added to PATH.
- `git` installed and added to PATH. See [instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
- `ffmpeg` installed and added to PATH. See instructions for [Windows](https://phoenixnap.com/kb/ffmpeg-windows), [Linux](https://phoenixnap.com/kb/install-ffmpeg-ubuntu) or [macOS](https://phoenixnap.com/kb/ffmpeg-mac).

Optionally, to use Nvidia GPU on Windows:
- CUDA version ≥12.0. Install from [Nvidia's official site](https://developer.nvidia.com/cuda-downloads).

>Note: For AMD GPUs (ROCm), GPU support for Whisper is only available in Linux.

## Set up
- In **Windows**, run the `whisper-gui.bat` file. In **Linux / macOS** run the `whisper-gui.sh` file. Follow the instructions and let the script install the necessary dependencies. After the process, it will run the GUI in a new browser tab.

Otherwise, manual steps are:
- **Create a conda environment with Python 3.10**  
	`conda create --name whisperx python=3.10`  
	`conda activate whisperx`
- **Install PyTorch 2.0**  
	For macOS:  
	`conda install pytorch::pytorch==2.0.0 torchaudio==2.0.0 -c pytorch`  
	For Windows or Linux, if you have Nvidia GPU:  
	`conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia`  
	For Linux, if you have AMD GPU:  
	`pip install torch==2.0.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/rocm6.0`  
	If not, install for CPU:  
	`conda install pytorch==2.0.0 torchaudio==2.0.0 cpuonly -c pytorch`
- **Install whisperx and dependecies**  
	`pip install git+https://github.com/m-bain/whisperx.git`  
	Original instructions in: https://github.com/m-bain/whisperX
- **Install additional libraries**  
	`pip install gradio`  
- **Clone this repository**  
	`git clone https://github.com/Pikurrot/whisper-gui`

## Run the GUI
To run the program every time, you can just run the same `whisper-gui.bat` or `whisper-gui.sh` (whatever your OS), which will also automatically check for updates of this repository.  
Your transcriptions will be saved by default in the `outputs` folder of the repository.

Otherwise, to run manually:  
`conda activate whisperx`  
`python main.py --autolaunch`

## Docker container (CPU only for now)
To run this software in a docker container, visit this [dockerhub project](https://hub.docker.com/r/3x3cut0r/whisper-gui).  
Thank you [3x3cut0r](https://hub.docker.com/u/3x3cut0r)!

## Licensing
This project is primarily distributed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

**Third-Party Code**  
Portions of this project incorporate code from [WhisperX](https://github.com/m-bain/whisperX), which is licensed under BSD-4-Clause license. This code is used in accordance with its license, and the full text of the license can be found within the relevant [source files](scripts/whisper_model.py).
