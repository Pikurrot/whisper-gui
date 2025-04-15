import os
import sys

def blockPrint():
	sys.stdout = open(os.devnull, 'w')
	# sys.stderr = open(os.devnull, 'w')
def enablePrint():
	sys.stdout = sys.__stdout__
	# sys.stderr = sys.__stderr__

blockPrint()
import gradio as gr
import torch
import whisperx
import gc
import argparse
import inspect
import time
import json
import subprocess
enablePrint()
from scripts.whisper_model import load_custom_model, LANG_CODES
from typing import Optional, Tuple, Callable
from scripts.config_io import read_config_value, write_config_value
from scripts.utils import *  # noqa: F403

# ensure gpu_support has correct value
gpu_support, error = read_config_value("gpu_support")
if gpu_support is False:
	write_config_value("gpu_support", "false")
	gpu_support = "false"
if error or gpu_support not in ("false", "cuda", "rocm", "mps"):
	# Check for Apple Silicon MPS
	if torch.backends.mps.is_available():
		write_config_value("gpu_support", "mps")
	# Check for NVIDIA GPU
	elif sys.platform != "darwin":  # Skip nvidia-smi check on macOS
		try:
			result = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			if result.returncode == 0:
				write_config_value("gpu_support", "cuda")
			else:
				result = subprocess.run("lspci | grep -i 'amdgpu'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				if result.returncode == 0:
					write_config_value("gpu_support", "rocm")
				else:
					write_config_value("gpu_support", "false")
		except FileNotFoundError:
			write_config_value("gpu_support", "false")
	else:
		write_config_value("gpu_support", "false")

# global variables
ALIGN_LANGS = ["en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl", "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca", "ml", "no", "nn"]
g_model = None
g_model_a = None
g_model_a_metadata = None
g_params = {}
with open("configs/lang.json", "r", encoding="utf-8") as f:
	LANG_DICT = reformat_lang_dict(json.load(f))
val, error = read_config_value("language")
if error:
	write_config_value("language", "en")
	LANG = "en"
else:
	LANG = val
if LANG not in LANG_DICT:
	LANG = "en"
	print(f"WARNING! Language {LANG} not supported for the interface. Using English instead")
MSG: dict[str, str] = LANG_DICT[LANG]

def release_whisper():
	"""
	Release the whisper model from memory.
	"""
	global g_model, g_params
	del g_model
	if g_params.get("device", None) == "gpu":
		torch.cuda.empty_cache()
	else:
		gc.collect()
	g_model = None
	print(MSG["whisper_released"])

def release_align():
	"""
	Release the alignment model from memory.
	"""
	global g_model_a, g_params
	del g_model_a
	if g_params.get("device", None) == "gpu":
		torch.cuda.empty_cache()
	else:
		gc.collect()
	g_model_a = None
	print(MSG["align_released"])

def release_memory_models():
	"""
	Release both models from memory.
	"""
	global g_model, g_model_a, g_params
	del g_model, g_model_a
	if g_params.get("device", None) == "gpu":
		torch.cuda.empty_cache()
	else:
		gc.collect()
	g_model = None
	g_model_a = None
	print(MSG["both_released"])

def get_args_str(func: Callable) -> list:
	"""
	Get the names of the arguments of a function.
	"""
	return list(inspect.signature(func).parameters)

def get_params(
		func: Callable,
		values: list
) -> dict:
	"""
	Get the parameters of a function as a dictionary.
	"""
	keys = get_args_str(func)
	return {k: values[k] for k in keys}

def same_params(
		params1: dict,
		params2: dict,
		*args
) -> bool:
	"""
	Check if two sets of parameters are the same.
	If args are provided, only check the specified parameters.
	"""
	if args:
		return all(params1.get(arg, None) == params2.get(arg, None) for arg in args)
	else:
		return params1 == params2

def transcribe_whisperx(
		model_name: str,
		audio_path: str,
		micro_audio: tuple,
		device: str,
		batch_size: int,
		compute_type: str,
		language: str,
		chunk_size: int,
		beam_size: int,
		release_memory: bool,
		save_root: Optional[str],
		save_audio: bool,
		save_transcription: bool,
		save_alignments: bool,
		save_in_subfolder: bool,
		preserve_name: bool,
		alignments_format: str
) -> Tuple[str, str, str, str]:
	"""
	Transcribe an audio file using the WhisperX model.
		Returns the transcription and sentence-level alignments.
	"""
	print(MSG["inputs_received"])
	if device == "gpu":
		device = "cuda"
	params = get_params(transcribe_whisperx, locals())
	global g_model, g_params

	if not same_params(params, g_params, "language"):
		print(MSG["lang_changed"])
		release_align()

	if not same_params(params, g_params, "model_name", "device", "compute_type", "beam_size") or g_model is None:
		if g_model is not None:
			print(MSG["params_changed"])
			release_whisper()
		print(MSG["loading_model"])
		blockPrint()
		g_model = whisperx.load_model(model_name, device, compute_type=compute_type, asr_options={"beam_size": beam_size}, download_root="models/whisperx")
		enablePrint()
	g_params = params

	return _transcribe()

def transcribe_custom(
		model_name: str,
		audio_path: str,
		micro_audio: tuple,
		device: str,
		batch_size: int,
		compute_type: str,
		language: str,
		chunk_size: int,
		beam_size: int,
		release_memory: bool,
		save_root: Optional[str],
		save_audio: bool,
		save_transcription: bool,
		save_alignments: bool,
		save_in_subfolder: bool,
		preserve_name: bool,
		alignments_format: str
) -> Tuple[str, str, str, str]:
	"""
	Transcribe an audio file using a custom Whisper model.
		Returns the transcription and sentence-level alignments.
	"""
	print(MSG["inputs_received"])
	if device == "gpu":
		device = "cuda"
	params = get_params(transcribe_custom, locals())
	global g_model, g_params

	if not same_params(params, g_params, "language", "device"):
		print(MSG["lang_changed"])
		release_align()

	if not same_params(params, g_params, "model_name", "device", "compute_type", "beam_size") or g_model is None:
		if g_model is not None:
			print(MSG["params_changed"])
			release_memory_models()
		print(MSG["loading_model"])
		blockPrint()
		g_model = load_custom_model(model_name, device, compute_type=compute_type, beam_size=beam_size, download_root="models/custom")
		enablePrint()
	g_params = params

	return _transcribe()

def _transcribe() -> Tuple[str, str, str, str]:
	"""
	Transcribe the audio file using the Whisper model.
	Models and parameters should be loaded and stored globally before calling this function.
		Returns the transcription and sentence-level alignments.
	"""
	global g_model, g_model_a, g_model_a_metadata, g_params
	# Create save folder
	save_dir = None
	temp_dir = os.path.join("temp", str(int(time.time())))  # Use timestamp for temp dir
	os.makedirs(temp_dir, exist_ok=True)
	
	if g_params["save_audio"] or g_params["save_transcription"] or g_params["save_alignments"]:
		if g_params["save_root"] is not None and g_params["save_root"] != "":
			save_root = g_params["save_root"]
		else:
			save_root = "outputs"
		if g_params["save_in_subfolder"]:
			save_dir = create_save_folder(save_root)
		else:
			save_dir = save_root

	try:
		# Load (and save) audio
		audio = load_and_save_audio(g_params["audio_path"], g_params["micro_audio"], g_params["save_audio"], save_dir, g_params["preserve_name"])

		# Transcription
		if g_params["language"] == "auto": 
			language = None
		else:
			language = g_params["language"]
		time_transcribe = time.time()
		print(MSG["starting_transcription"])
		result = g_model.transcribe(audio, batch_size=g_params["batch_size"], language=language, chunk_size=g_params["chunk_size"], print_progress=True)
		if "time" in result.keys():
			time_transcribe = result["time"]
		else:
			time_transcribe = time.time() - time_transcribe
		joined_text = " ".join([segment["text"].strip() for segment in result["segments"]])
		if g_params["save_transcription"]:
			if g_params["preserve_name"]:
				audio_name = os.path.basename(g_params["audio_path"]).split(".")[0]
				save_name = f"{audio_name}_transcription.txt"
			else:
				save_name = "transcription.txt"
			save_transcription_to_txt(joined_text, save_dir, save_name)

		if g_params["release_memory"]:
			release_whisper()

		# Word-level alignment
		lang_used = result["language"]
		if lang_used not in ALIGN_LANGS:
			print(MSG["align_lang_not_supported"].format(lang_used))
			lang_used = "en"
		if g_model_a is None:
			print(MSG["loading_align_model"])
			g_model_a, g_model_a_metadata = whisperx.load_align_model(language_code=lang_used, device=g_params["device"], model_dir="models/alignment")
		print(MSG["aligning"])
		time_align = time.time()
		aligned_result = whisperx.align(result["segments"], g_model_a, g_model_a_metadata, audio, g_params["device"], return_char_alignments=False)
		time_align = time.time() - time_align
		if g_params["save_alignments"]:
			align_format = g_params["alignments_format"].lower()
			if g_params["preserve_name"]:
				audio_name = os.path.basename(g_params["audio_path"]).split(".")[0]
				save_name = f"{audio_name}_timestamps." + align_format
			else:
				save_name = "timestamps." + align_format
			if align_format == "json":
				save_alignments_to_json(aligned_result, save_dir, save_name)
			elif align_format == "srt":
				subtitles = alignments2subtitles(aligned_result["segments"], max_line_length=50)
				save_subtitles_to_srt(subtitles, save_dir, save_name)
		if g_params["release_memory"]:
			release_align()
		print(MSG["done"])
		
		return joined_text, format_alignments(aligned_result), f"{round(time_transcribe, 3)}s", f"{round(time_align, 3)}s"
	finally:
		# Clean up temp directory
		try:
			import shutil
			if os.path.exists(temp_dir):
				shutil.rmtree(temp_dir)
		except Exception as e:
			print(f"Warning: Could not clean up temp directory: {e}")


# Prepare interface data
whisperx_models = ["large-v3", "large-v2", "large-v1", "medium", "small", "base", "tiny", "medium.en", "small.en", "base.en", "tiny.en"]
custom_models = list_models()
whisperx_langs = ["auto", "en", "es", "fr", "de", "it", "ja", "zh", "nl", "uk", "pt"]
custom_langs = ["auto"] + list(LANG_CODES.keys())

# Read config
gpu_support, error = read_config_value("gpu_support")
if gpu_support in ("cuda", "rocm"):
	device = "gpu"
	device_interactive = True
	device_message = ""
else:
	device = "cpu"
	device_interactive = False
	if gpu_support is None:
		device_message = MSG["select_cpu"]
	else:
		device_message = MSG["gpu_disabled"]

def apply_config(lang: str):
	prev_lang, error = read_config_value("language")
	prev_lang = prev_lang if not error else LANG
	write_config_value("language", lang)
	if lang != prev_lang:
		gr.Info(MSG["settings_updated"])

# Gradio interface
with gr.Blocks(title="Whisper GUI") as demo:
	gr.Markdown(f"""# Whisper GUI
{MSG["gui_description"]}""")
	with gr.Tab("Faster Whisper"):
		with gr.Row():
			with gr.Column():
				model_select = gr.Dropdown(whisperx_models, value="base", label=MSG["model_select_label"], info=MSG["change_whisper_reload"])
				with gr.Group():
					file_upload = gr.File(
						label="Upload Audio/Video File",
						file_types=[".mp3", ".wav", ".m4a", ".mp4", ".avi", ".mov", ".mkv", ".webm"],
						type="filepath"
					)
					audio_record = gr.Audio(sources=["microphone"], type="numpy", label=MSG["audio_record_label"], visible=False)
					save_audio = gr.Checkbox(value=False, label="Save extracted audio", info="Save the audio/extracted audio to the output directory")
				gr.Examples(examples=["examples/coffe_break_example.mp3"], inputs=file_upload)
				with gr.Accordion(label=MSG["advanced_options"], open=False):
					language_select = gr.Dropdown(whisperx_langs, value = "auto", label=MSG["language_select_label"], info=MSG["language_select_info"]+MSG["change_align_reload"])
					device_select = gr.Radio(["gpu", "cpu"], value = device, label=MSG["device_select_label"], info=device_message+MSG["change_both_reload"], interactive=device_interactive)
					with gr.Group():
						with gr.Row():
							save_transcription = gr.Checkbox(value=True, label=MSG["save_transcription_label"])
							save_alignments = gr.Checkbox(value=True, label=MSG["save_align_label"])
						save_root = gr.Textbox(label=MSG["save_root_label"], placeholder="outputs", lines=1)
						save_in_subfolder = gr.Checkbox(value=True, label=MSG["save_subfolder_label"], info=MSG["save_subfolder_info"])
						preserve_name = gr.Checkbox(value=False, label=MSG["preserve_name_label"], info=MSG["preserve_name_info"])
						alignments_format = gr.Radio(["JSON", "SRT"], value="JSON", label=MSG["align_format_label"], interactive=True)
					gr.Markdown(f"""### {MSG["optimizations"]}""")
					compute_type_select = gr.Radio(["int8", "float16", "float32"], value = "int8", label=MSG["compute_type_label"], info=MSG["compute_type_info"]+MSG["change_whisper_reload"])
					batch_size_slider = gr.Slider(1, 128, value = 1, step=1, label=MSG["batch_size_label"], info=MSG["batch_size_info"])
					chunk_size_slider = gr.Slider(1, 80, value = 20, step=1, label=MSG["chunk_size_label"], info=MSG["chunk_size_info"])
					beam_size_slider = gr.Slider(1, 100, value = 5, step=1, label=MSG["beam_size_label"], info=MSG["beam_size_info"]+MSG["change_whisper_reload"])
					release_memory_checkbox = gr.Checkbox(label=MSG["release_memory_label"], value=True, info=MSG["release_memory_info"])
				submit_button = gr.Button(value=MSG["submit_button"])
			with gr.Column():
				transcription_output = gr.Textbox(label=MSG["transcription_textbox"], lines=15)
				alignments_output = gr.Textbox(label=MSG["align_textbox"], lines=15)
				with gr.Row():
					time_transcribe = gr.Textbox(label=MSG["time_transcribe_label"], info=MSG["time_transcribe_info"], lines=1)
					time_align = gr.Textbox(label=MSG["time_align_label"], lines=1)
				release_memory_button = gr.Button(value=MSG["release_memory_button"])

	with gr.Tab("Custom model"):
		with gr.Row():
			with gr.Column():
				with gr.Group():
					model_select2 = gr.Dropdown(custom_models, value=None, label=MSG["model_select2_label"], allow_custom_value=True, info=MSG["change_whisper_reload"])
				with gr.Group():
					file_upload2 = gr.File(
						label="Upload Audio/Video File",
						file_types=[".mp3", ".wav", ".m4a", ".mp4", ".avi", ".mov", ".mkv", ".webm"],
						type="filepath"
					)
					audio_record2 = gr.Audio(sources=["microphone"], type="numpy", label=MSG["audio_record_label"], visible=False)
					save_audio2 = gr.Checkbox(value=False, label="Save extracted audio", info="Save the audio/extracted audio to the output directory")
				gr.Examples(examples=["examples/coffe_break_example.mp3"], inputs=file_upload2)
				with gr.Accordion(label=MSG["advanced_options"], open=False):
					language_select2 = gr.Dropdown(custom_langs, value = "auto", label="Language", info=MSG["language_select_info"]+MSG["change_align_reload"])
					device_select2 = gr.Radio(["gpu", "cpu"], value = device, label=MSG["device_select_label"], info=device_message+MSG["change_both_reload"], interactive=device_interactive)
					with gr.Group():
						with gr.Row():
							save_transcription2 = gr.Checkbox(value=True, label=MSG["save_transcription_label"])
							save_alignments2 = gr.Checkbox(value=True, label=MSG["save_align_label"])
						save_root2 = gr.Textbox(label=MSG["save_root_label"], placeholder="outputs", lines=1)
						save_in_subfolder2 = gr.Checkbox(value=True, label=MSG["save_subfolder_label"], info=MSG["save_subfolder_info"])
						preserve_name2 = gr.Checkbox(value=False, label=MSG["preserve_name_label"], info=MSG["preserve_name_info"])
						alignments_format2 = gr.Radio(["JSON", "SRT"], value="JSON", label=MSG["align_format_label"], interactive=True)
					gr.Markdown(f"""### {MSG["optimizations"]}""")
					compute_type_select2 = gr.Radio(["float16", "float32"], value = "float16", label=MSG["compute_type_label"], info=MSG["compute_type_info"]+MSG["change_whisper_reload"])
					batch_size_slider2 = gr.Slider(1, 128, value = 1, step=1, label=MSG["batch_size_label"], info=MSG["batch_size_info"])
					chunk_size_slider2 = gr.Slider(1, 80, value = 20, step=1, label=MSG["chunk_size_label"], info=MSG["chunk_size_info"])
					beam_size_slider2 = gr.Slider(1, 100, value = 5, step=1, label=MSG["beam_size_label"], info=MSG["beam_size_info"]+MSG["change_whisper_reload"])
					release_memory_checkbox2 = gr.Checkbox(label=MSG["release_memory_label"], value=True, info=MSG["release_memory_info"])
				submit_button2 = gr.Button(value=MSG["submit_button"])
			with gr.Column():
				transcription_output2 = gr.Textbox(label=MSG["transcription_textbox"], lines=15)
				alignments_output2 = gr.Textbox(label=MSG["align_textbox"], lines=15)
				with gr.Row():
					time_transcribe2 = gr.Textbox(label=MSG["time_transcribe_label"], info=MSG["time_transcribe_info"], lines=1)
					time_align2 = gr.Textbox(label=MSG["time_align_label"], lines=1)
				release_memory_button2 = gr.Button(value=MSG["release_memory_button"])

	with gr.Tab("Settings"):
		lang_select = gr.Dropdown(LANG_DICT.keys(), value=LANG, label=MSG["lang_select_label"], allow_custom_value=True, info=MSG["lang_select_info"])
		apply_button = gr.Button(value=MSG["apply_changes"])
	
	submit_button.click(transcribe_whisperx,
						inputs=[model_select, file_upload, audio_record, device_select, batch_size_slider, compute_type_select, language_select, chunk_size_slider, beam_size_slider, release_memory_checkbox, save_root, save_audio, save_transcription, save_alignments, save_in_subfolder, preserve_name, alignments_format],
						outputs=[transcription_output, alignments_output, time_transcribe, time_align])
	
	submit_button2.click(transcribe_custom,
						inputs=[model_select2, file_upload2, audio_record2, device_select2, batch_size_slider2, compute_type_select2, language_select2, chunk_size_slider2, beam_size_slider2, release_memory_checkbox2, save_root2, save_audio2, save_transcription2, save_alignments2, save_in_subfolder2, preserve_name2, alignments_format2],
						outputs=[transcription_output2, alignments_output2, time_transcribe2, time_align2])
	
	release_memory_button.click(release_memory_models)
	release_memory_button2.click(release_memory_models)

	apply_button.click(apply_config, inputs=[lang_select])


if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description=MSG["argparse_description"])
	parser.add_argument("--autolaunch", action="store_true", default=False, help=MSG["autloaunch_help"])
	parser.add_argument("--share", action="store_true", default=False, help=MSG["share_help"])
	args = parser.parse_args()

	# Launch the interface
	print(MSG["creating_interface"])
	# When running in Docker, we need to bind to 0.0.0.0
	is_docker = os.path.exists('/.dockerenv')
	demo.launch(
		inbrowser=args.autolaunch,
		share=args.share,
		server_name='0.0.0.0' if is_docker else None
	)
