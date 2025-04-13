#!/bin/bash

# Variables
CONFIG_FILE="$(dirname "$0")/configs/config.json"
ENV_NAME=""
GPU_SUPPORT=""
AUTO_UPDATE=""
ENV_COUNT=0
TEMP=""
PYTHON_VERSION=""
DEP_FILE=""
LOCAL_COMMIT=""
REMOTE_COMMIT=""
# Initialize list of conda environments
MAX_ENVS=20
declare -a envs

start() {
	# Check conda is installed
	echo "Checking conda installation..."
	if ! conda --version &> /dev/null; then
		echo "conda is not installed. Install Anaconda first and add conda command to PATH."
		exit 1
	fi
	echo "conda is installed."

	# Check if config file exists
	echo "Checking config file..."
	if [[ -f "$CONFIG_FILE" ]]; then
		read_env_name
	else
		echo "Config file does not exist. Creating config file..."
		touch "$CONFIG_FILE"
		echo "Config file created."
		list_envs
	fi
}

# List conda environments
list_envs() {
	echo "Listing conda environments..."
	while IFS= read -r line; do
	if [[ -n "$line" ]]; then # Skip empty lines
		envs[ENV_COUNT]="$line"
		echo "$ENV_COUNT. $line"
		((ENV_COUNT++))
	fi
	done < <(conda env list | awk 'NR > 2 && $1 != "" {print $1}')
	
	if [[ $ENV_COUNT -eq 0 ]]; then
		echo "No conda environments found."
		read -p "Do you want to create a new environment named 'whisperx'? ([y]/n): " TEMP
		if [[ "$TEMP" == "n" ]]; then
			echo "Exiting..."
			exit 0
		fi
		ENV_NAME="whisperx"
		create_env
	else
		echo "$ENV_COUNT. Create a new environment (RECOMMENDED)"
		echo
		select_env
	fi
}

select_env() {
	read -p "Select the number of the environment to use: " TEMP
	if [[ $TEMP -lt 0 ]] || [[ $TEMP -gt $ENV_COUNT ]]; then
		echo "Invalid input."
		select_env
		return
	fi
	if [[ "$TEMP" -eq "$ENV_COUNT" ]]; then
		read -p "Enter the name of the new environment: " ENV_NAME
		create_env
		return
	fi
	
	ENV_NAME="${envs[$TEMP]}"
	echo "Selected environment: '$ENV_NAME'"
	
	# Check Python version
	echo "Checking Python version..."
	conda_base=$(dirname $(dirname $CONDA_EXE))
    source "$conda_base/etc/profile.d/conda.sh"
	conda activate "$ENV_NAME" &> /dev/null
	PYTHON_VERSION=$(python --version 2>&1)
	if [ $? -ne 0 ]; then
		echo "Python is not installed in this environment. Install Python first."
		conda deactivate &> /dev/null
		exit 1
	fi
	PYTHON_MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d ' ' -f 2 | cut -d '.' -f 1)
	PYTHON_MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d ' ' -f 2 | cut -d '.' -f 2)
	
	if [ $PYTHON_MAJOR_VERSION -eq 3 ] && [ $PYTHON_MINOR_VERSION -lt 10 ]; then
		echo "Incorrect Python version. Install Python 3.10 or newer."
		conda deactivate &> /dev/null
		exit 1
	fi
	
	echo "Python version is correct: $PYTHON_VERSION"
	save_env_name
}

create_env() {
	echo "Creating new environment '$ENV_NAME' with python 3.10.13..."
	if ! conda create -n "$ENV_NAME" python=3.10.13 -y &> /dev/null; then
		echo "Failed to create environment."
		exit 1
	fi
	echo "Environment created."
	activate_env
}

read_env_name() {
	ENV_NAME=$(scripts/config_read.sh "env_name")
	TEMP=$?
	
	if [[ $TEMP -eq 1 ]]; then
		echo "Failed to read config file."
		exit 1
	elif [[ $TEMP -eq 2 ]]; then
		echo "Config file does not contain 'env_name' key."
		list_envs
		return
	elif [[ $TEMP -eq 3 ]]; then
		echo "'env_name' key is null in config file."
		list_envs
		return
	fi
	
	TEMP="skip"
	activate_env
}

activate_env() {
	echo "Activating environment $ENV_NAME..."
	conda_base=$(dirname $(dirname $CONDA_EXE))
    source "$conda_base/etc/profile.d/conda.sh"
	conda activate "$ENV_NAME"
	if [[ $? -ne 0 ]]; then
		echo "Failed to activate environment."
		exit 1
	fi
	echo "Environment activated."
	if [[ -n $TEMP && $TEMP == "skip" ]]; then
		check_updates
	else
		save_env_name
	fi
}

save_env_name() {
	echo "Saving environment name to config file..."
	if ! python scripts/config_write.py "env_name" "$ENV_NAME" &> /dev/null; then
		echo "Failed to save environment name to config file."
		exit 1
	fi
	check_gpu
}

check_gpu() {
	echo "Checking GPU support..."
	GPU_SUPPORT=$(scripts/config_read.sh "gpu_support")
	TEMP=$?

	if [[ $TEMP -eq 0 ]]; then
		install_deps
	elif [[ $TEMP -eq 1 ]]; then
		echo "Failed to read config file."
		exit 1
	elif [[ $TEMP -eq 2 ]]; then
		echo "Config file does not contain 'gpu_support' key."
		test_gpu
	elif [[ $TEMP -eq 3 ]]; then
		echo "'gpu_support' key is null in config file."
		test_gpu
	fi
}

test_gpu() {
	echo "Testing if GPU is available..."
	GPU_SUPPORT="false"
	if nvidia-smi &> /dev/null; then
		echo "NVIDIA GPU detected."
		read -p "Do you want to use GPU support? ([y]/n): " TEMP
		if [[ "$TEMP" == "y" ]]; then
			GPU_SUPPORT="cuda"
			echo "CUDA GPU support enabled."
		else
			echo "Proceeding with CPU support."
		fi
	elif lspci | grep -i 'amdgpu' &> /dev/null; then
		echo "AMD GPU detected."
		read -p "Do you want to use GPU support? ([y]/n): " TEMP
		if [[ "$TEMP" == "y" ]]; then
			GPU_SUPPORT="rocm"
			echo "ROCm GPU support enabled."
		else
			echo "Proceeding with CPU support."
		fi
	else
		echo "No GPU detected. Proceeding with CPU support."
	fi
	echo "Saving result to config file..."
	if ! python scripts/config_write.py "gpu_support" "$GPU_SUPPORT" &> /dev/null; then
		echo "Failed to save result to config file."
		exit 1
	fi
	install_deps
}

install_deps() {
	TEMP=$(uname -s)
	echo "Operating system: $TEMP"

	if [ "$TEMP" == "Linux" ]; then
		if [[ "$GPU_SUPPORT" == "cuda" ]]; then
			echo "Installing dependencies for CUDA GPU..."
			DEP_FILE="configs/environment_gpu.yml"
			echo "Installing PyTorch with CUDA 11.8..."
			pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
		elif [[ "$GPU_SUPPORT" == "rocm" ]]; then
			echo "Installing dependencies for ROCm GPU..."
			DEP_FILE="configs/environment_cpu.yml"
			echo "Installing PyTorch with ROCm..."
			pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
		else
			echo "Installing dependencies for CPU..."
			DEP_FILE="configs/environment_cpu.yml"
			pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
		fi
	elif [ "$TEMP" == "Darwin" ]; then
		if [[ "$GPU_SUPPORT" == "rocm" ]]; then
			echo "ROCm is not supported on macOS. Proceeding with CPU support."
		fi
		echo "Installing dependencies for macOS..."
		DEP_FILE="configs/environment_cpu.yml"
		pip install torch torchaudio
	else
		echo "This is neither Linux nor macOS."
		exit 1
	fi

	echo "Installing whisperx..."
	pip install git+https://github.com/m-bain/whisperx.git
	echo "Installing other dependencies..."
	conda env update --name "$ENV_NAME" --file "$DEP_FILE"
	if [[ $? -ne 0 ]]; then
		echo "Failed to install dependencies."
		exit 1
	fi
	echo "Dependencies installed."
	if [[ -n $LOCAL_COMMIT ]]; then
		set_auto_update
	else
		check_updates
	fi
}

check_updates() {
	echo "Checking for updates..."
	if ! git --version &> /dev/null; then
		echo "git is not installed. Install git first and add git command to PATH."
		run
		return
	fi

	if ! git fetch &> /dev/null; then
		echo "Failed to fetch updates. Check your internet connection."
		run
		return
	fi

	LOCAL_COMMIT=$(git rev-parse master)
	REMOTE_COMMIT=$(git rev-parse origin/master)

	AUTO_UPDATE=$(scripts/config_read.sh "auto_update")
	if [[ $? -eq 1 ]]; then
		echo "Failed to read config file."
		run
		return
	fi

	if [[ "$LOCAL_COMMIT" == "$REMOTE_COMMIT" ]]; then
		echo "Your repository is up to date."
		set_auto_update
		return
	else
		echo "Updates available."
		if [[ "$AUTO_UPDATE" != "true" ]]; then
			read -p "Do you want to update the repository? ([y]/n): " TEMP
			if [[ "$TEMP" == "n" ]]; then
				set_auto_update
				return
			fi
		fi
		update_repo
	fi
}

update_repo() {
	echo "Updating repository..."
	if ! git pull origin master &> /dev/null; then
		echo "Failed to update repository."
		set_auto_update
	else
		if [[ -n $GPU_SUPPORT ]]; then
			install_deps
		else
			check_gpu
		fi
	fi
}

set_auto_update() {
	echo "Checking auto update setting..."
	if [[ -z $AUTO_UPDATE ]]; then
		read -p "Do you want this repository to update automatically from now on? ([y]/n): " TEMP
		if [[ "$TEMP" == "n" ]]; then
			AUTO_UPDATE=false
			echo "You will be asked to update this repository every time an update is available."
			# TODO: You can change this setting in the configuration later.
		else
			AUTO_UPDATE=true
			echo "This repository will update automatically from now on."
		fi
		echo "Saving result to config file..."
		if ! python scripts/config_write.py "auto_update" "$AUTO_UPDATE" &> /dev/null; then
			echo "Failed to save result to config file."
		fi
	fi
	run
}

run() {
	echo "Running Whisper-GUI..."
	python main.py --autolaunch
}

start
