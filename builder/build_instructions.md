# Building an Integrated Package (only admins)
How to build a single executable file of this project for each OS. This will include the software source code and dependencies. Moreover, a different executable will be created for CPU-only, CUDA.

## For Windows
### CUDA
```bash
cd builder
conda activate whisperx_temp
```
```bash
pyinstaller ^
--paths .. ^
--icon=icon_multi_res.ico ^
--hidden-import=main ^
--hidden-import=whisperx ^
--collect-all gradio ^
--collect-all gradio_client ^
--collect-all safehttpx ^
--collect-all groovy ^
--collect-submodules whisperx ^
--collect-all whisperx ^
--collect-all lightning_fabric ^
--collect-all pyannote ^
--collect-all speechbrain ^
--add-data ../configs:configs ^
--add-data ../examples:examples ^
--workpath build ^
--distpath dist ^
--specpath . ^
--onefile ^
--name whisper-gui-cuda-win64 ^
_launcher_cuda.py
```
> Optional: remove `build/` once done to free space.

### CPU
```bash
cd builder
conda activate whisperx_cpu
```
```bash
pyinstaller ^
--paths .. ^
--icon=icon_multi_res.ico ^
--hidden-import=main ^
--hidden-import=whisperx ^
--collect-all gradio ^
--collect-all gradio_client ^
--collect-all safehttpx ^
--collect-all groovy ^
--collect-submodules whisperx ^
--collect-all whisperx ^
--collect-all lightning_fabric ^
--collect-all pyannote ^
--collect-all speechbrain ^
--add-data ../configs:configs ^
--add-data ../examples:examples ^
--workpath build ^
--distpath dist ^
--specpath . ^
--onefile ^
--name whisper-gui-cpu-win64 ^
_launcher_cpu.py
```
> Optional: remove `build/` once done to free space.

## For Linux
### CUDA
```bash
cd builder
conda activate whisperx_temp
```
```bash
pyinstaller \
--paths .. \
--hidden-import=main \
--hidden-import=whisperx \
--collect-all gradio \
--collect-all gradio_client \
--collect-all safehttpx \
--collect-all groovy \
--collect-submodules whisperx \
--collect-all whisperx \
--collect-all lightning_fabric \
--collect-all pyannote \
--collect-all speechbrain \
--add-data ../configs:configs \
--add-data ../examples:examples \
--workpath build \
--distpath dist \
--specpath . \
--onefile \
--name whisper-gui-cuda-linux \
_launcher_cuda.py
```
> Optional: remove `build/` once done to free space.

Then, to wrap in an appimage for better compatibility:
```bash
mkdir -p WhisperGUI-cuda.AppDir/usr/bin
mv dist/whisper-gui-cuda-linux WhisperGUI-cuda.AppDir/usr/bin/
ARCH=x86_64 appimagetool WhisperGUI-cuda.AppDir
```

### CPU
```bash
cd builder
conda activate whisperx_cpu
```
```bash
pyinstaller \
--paths .. \
--hidden-import=main \
--hidden-import=whisperx \
--collect-all gradio \
--collect-all gradio_client \
--collect-all safehttpx \
--collect-all groovy \
--collect-submodules whisperx \
--collect-all whisperx \
--collect-all lightning_fabric \
--collect-all pyannote \
--collect-all speechbrain \
--add-data ../configs:configs \
--add-data ../examples:examples \
--workpath build \
--distpath dist \
--specpath . \
--onefile \
--name whisper-gui-cpu-linux \
_launcher_cpu.py
```
> Optional: remove `build/` once done to free space.

Then, to wrap in an appimage for better compatibility:
```bash
mkdir -p WhisperGUI-cpu.AppDir/usr/bin
mv dist/whisper-gui-cpu-linux WhisperGUI-cpu.AppDir/usr/bin/
ARCH=x86_64 appimagetool WhisperGUI-cpu.AppDir
```
