import os
import sys

def blockPrint():
	sys.stdout = open(os.devnull, 'w')
	sys.stderr = open(os.devnull, 'w')
def enablePrint():
	sys.stdout = sys.__stdout__
	sys.stderr = sys.__stderr__

blockPrint()
import gradio as gr
import whisperx
import gc
import argparse
import inspect
import torch
import time
from scripts.whisper_model import load_custom_model, LANG_CODES
from typing import Optional, Tuple, Callable
from scripts.config_io import read_config_value
from utils import alignments2subtitles, create_save_folder, format_alignments, list_models, load_and_save_audio, save_alignments_to_json, save_subtitles_to_srt, save_transcription_to_txt

enablePrint()
ALIGN_LANGS = ["en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl", "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca", "ml", "no", "nn"]

# global variables
g_model = None
g_model_a = None
g_model_a_metadata = None
g_params = {}

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
	print("Whisper model released from memory")

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
	print("Alignment model released from memory")

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
	print("Models released from memory")

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
	print("Inputs received. Starting...")
	if device == "gpu":
		device = "cuda"
	params = get_params(transcribe_whisperx, locals())
	global g_model, g_params

	if not same_params(params, g_params, "language"):
		print("Language changed. Releasing previous alignment model from memory...")
		release_align()

	if not same_params(params, g_params, "model_name", "device", "compute_type", "beam_size") or g_model is None:
		if g_model is not None:
			print("Parameters changed. Releasing previous whisper model from memory...")
			release_whisper()
		print("Loading model...")
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
	print("Inputs received. Starting...")
	if device == "gpu":
		device = "cuda"
	params = get_params(transcribe_custom, locals())
	global g_model, g_params

	if not same_params(params, g_params, "language", "device"):
		print("Language changed. Releasing previous alignment model from memory...")
		release_align()

	if not same_params(params, g_params, "model_name", "device", "compute_type", "beam_size") or g_model is None:
		if g_model is not None:
			print("Parameters changed. Releasing previous models from memory...")
			release_memory_models()
		print("Loading model...")
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
	if not os.path.exists("temp"):
		os.makedirs("temp")
	if g_params["save_audio"] or g_params["save_transcription"] or g_params["save_alignments"]:
		if g_params["save_root"] is not None and g_params["save_root"] != "":
			save_root = g_params["save_root"]
		else:
			save_root = "outputs"
		if g_params["save_in_subfolder"]:
			save_dir = create_save_folder(save_root)
		else:
			save_dir = save_root

	# Load (and save) audio
	audio = load_and_save_audio(g_params["audio_path"], g_params["micro_audio"], g_params["save_audio"], save_dir, g_params["preserve_name"])

	# Transcription
	if g_params["language"] == "auto": 
		language = None
	else:
		language = g_params["language"]
	time_transcribe = time.time()
	print("Starting transcription...")
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
		print(f"WARNING! Language {lang_used} not supported for alignment. Using English instead. Results may be inaccurate.")
		lang_used = "en"
	if g_model_a is None:
		print("Loading alignment model...")
		g_model_a, g_model_a_metadata = whisperx.load_align_model(language_code=lang_used, device=g_params["device"], model_dir="models/alignment")
	print("Aligning...")
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
	print("Done!")
	if not os.listdir("temp") and os.path.exists("temp"):
		# Remove temp folder if empty
		os.rmdir("temp")
	# Return the transcription and sentence-level alignments
	return joined_text, format_alignments(aligned_result), f"{round(time_transcribe, 3)}s", f"{round(time_align, 3)}s"


# Prepare interface data
whisperx_models = ["large-v3", "large-v2", "large-v1", "medium", "small", "base", "tiny", "medium.en", "small.en", "base.en", "tiny.en"]
custom_models = list_models()
whisperx_langs = ["auto", "en", "es", "fr", "de", "it", "ja", "zh", "nl", "uk", "pt"]
custom_langs = ["auto"] + list(LANG_CODES.keys())
release_whisper_message = "When changed, requires the Whisper model to reload."
release_align_message = "When changed, requires the alignment model to reload."
release_both_message = "When changed, requires both models to reload."

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
		device_message = "If you don\"t have a GPU, select \"cpu\".\n"
	else:
		device_message = "GPU support is disabled in the config file.\n"

# Gradio interface
with gr.Blocks(title="Whisper GUI") as demo:
	gr.Markdown("""# Whisper GUI
A simple interface to transcribe audio files using the Whisper model""")
	with gr.Tab("Faster Whisper"):
		with gr.Row():
			with gr.Column():
				model_select = gr.Dropdown(whisperx_models, value="base", label="Load WhisperX Model", info=release_whisper_message)
				with gr.Group():
					audio_upload = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio File")
					audio_record = gr.Audio(sources=["microphone"], type="numpy", label="or Record Audio (If both are provided, only microphone audio will be used)")
					save_audio = gr.Checkbox(value=False, label="Save Audio", info="Save copy of audio to \"Save Path\" folder.")
				gr.Examples(examples=["examples/coffe_break_example.mp3"], inputs=audio_upload)
				with gr.Accordion(label="Advanced Options", open=False):
					language_select = gr.Dropdown(whisperx_langs, value = "auto", label="Language", info="Select the language of the audio file. Select \"auto\" to automatically detect it. "+release_align_message)
					device_select = gr.Radio(["gpu", "cpu"], value = device, label="Device", info=device_message+release_both_message, interactive=device_interactive)
					with gr.Group():
						with gr.Row():
							save_transcription = gr.Checkbox(value=True, label="Save Transcription")
							save_alignments = gr.Checkbox(value=True, label="Save Timestamps")
						save_root = gr.Textbox(label="Save Path", placeholder="outputs", lines=1)
						save_in_subfolder = gr.Checkbox(value=True, label="Save in Subfolder", info="Save files in a subfolder \"YYYY-MM-DD/NNNN/\" in the \"Save Path\" folder. CAUTION: if unchecked, files may be overwritten.")
						preserve_name = gr.Checkbox(value=False, label="Preserve Name", info="Preserve the original name of the audio file when saving. E.g. \"<audio_name>_transcription.txt\". Only works for uploaded audio.")
						alignments_format = gr.Radio(["JSON", "SRT"], value="JSON", label="Alignments Format", interactive=True)
					gr.Markdown("""### Optimizations""")
					compute_type_select = gr.Radio(["int8", "float16", "float32"], value = "int8", label="Compute Type", info="int8 is fastest and requires less memory. float32 is more accurate (Your device may not support some data types). "+release_whisper_message)
					batch_size_slider = gr.Slider(1, 128, value = 1, step=1, label="Batch Size", info="Larger batch sizes may be faster but require more memory.")
					chunk_size_slider = gr.Slider(1, 80, value = 20, step=1, label="Chunk Size", info="Larger chunk sizes may be faster but require more memory.")
					beam_size_slider = gr.Slider(1, 100, value = 5, step=1, label="Beam Size", info="Larger beam sizes may be more accurate but require more memory and may decrease speed. "+release_whisper_message)
					release_memory_checkbox = gr.Checkbox(label="Release Memory", value=True, info="Release Whisper model from memory before loading alignment model. Then release alignment model. Prevents having both models in memory at the same time.")
				submit_button = gr.Button(value="Start Transcription")
			with gr.Column():
				transcription_output = gr.Textbox(label="Transcription", lines=15)
				alignments_output = gr.Textbox(label="Timestamps", lines=15)
				with gr.Row():
					time_transcribe = gr.Textbox(label="Transcription Time", info="Including language detection (if Language = \"auto\")", lines=1)
					time_align = gr.Textbox(label="Alignment Time", lines=1)
				release_memory_button = gr.Button(value="Release Models from Memory")

	with gr.Tab("Custom model"):
		with gr.Row():
			with gr.Column():
				with gr.Group():
					model_select2 = gr.Dropdown(custom_models, value=None, label="Upload Local Model  or  Download a Model from HuggingFace", allow_custom_value=True, info=release_whisper_message)
				with gr.Group():
					audio_upload2 = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio File")
					audio_record2 = gr.Audio(sources=["microphone"], type="numpy", label="or Record Audio (If both are provided, only microphone audio will be used)")
					save_audio2 = gr.Checkbox(value=False, label="Save Audio", info="Save copy of audio to \"Save Path\" folder.")
				gr.Examples(examples=["examples/coffe_break_example.mp3"], inputs=audio_upload2)
				with gr.Accordion(label="Advanced Options", open=False):
					language_select2 = gr.Dropdown(custom_langs, value = "auto", label="Language", info="Select the language of the audio file. Select \"auto\" to automatically detect it. "+release_align_message)
					device_select2 = gr.Radio(["gpu", "cpu"], value = device, label="Device", info=device_message+release_both_message, interactive=device_interactive)
					with gr.Group():
						with gr.Row():
							save_transcription2 = gr.Checkbox(value=True, label="Save Transcription")
							save_alignments2 = gr.Checkbox(value=True, label="Save Timestamps")
						save_root2 = gr.Textbox(label="Save Path", placeholder="outputs", lines=1)
						save_in_subfolder2 = gr.Checkbox(value=True, label="Save in Subfolder", info="Save files in a subfolder \"YYYY-MM-DD/NNNN/\" in the \"Save Path\" folder. CAUTION: if unchecked, files may be overwritten.")
						preserve_name2 = gr.Checkbox(value=False, label="Preserve Name", info="Preserve the original name of the audio file when saving. E.g. \"<audio_name>_transcription.txt\". Only works for uploaded audio.")
						alignments_format2 = gr.Radio(["JSON", "SRT"], value="JSON", label="Alignments Format", interactive=True)
					gr.Markdown("""### Optimizations""")
					compute_type_select2 = gr.Radio(["float16", "float32"], value = "float16", label="Compute Type", info="float16 is faster and requires less memory. float32 is more accurate (Your device may not support some data types). "+release_whisper_message)
					batch_size_slider2 = gr.Slider(1, 128, value = 1, step=1, label="Batch Size", info="Larger batch sizes may be faster but require more memory.")
					chunk_size_slider2 = gr.Slider(1, 80, value = 20, step=1, label="Chunk Size", info="Larger chunk sizes may be faster but require more memory.")
					beam_size_slider2 = gr.Slider(1, 100, value = 5, step=1, label="Beam Size", info="Larger beam sizes may be more accurate but require more memory and may decrease speed. "+release_whisper_message)
					release_memory_checkbox2 = gr.Checkbox(label="Release Memory", value=True, info="Release Whisper model from memory before loading alignment model. Then release alignment model. Prevents having both models in memory at the same time.")
				submit_button2 = gr.Button(value="Start Transcription")
			with gr.Column():
				transcription_output2 = gr.Textbox(label="Transcription", lines=15)
				alignments_output2 = gr.Textbox(label="Timestamps", lines=15)
				with gr.Row():
					time_transcribe2 = gr.Textbox(label="Transcription Time", lines=1)
					time_align2 = gr.Textbox(label="Alignment Time", lines=1)
				release_memory_button2 = gr.Button(value="Release Models from Memory")

	
	submit_button.click(transcribe_whisperx,
						inputs=[model_select, audio_upload, audio_record, device_select, batch_size_slider, compute_type_select, language_select, chunk_size_slider, beam_size_slider, release_memory_checkbox, save_root, save_audio, save_transcription, save_alignments, save_in_subfolder, preserve_name, alignments_format],
						outputs=[transcription_output, alignments_output, time_transcribe, time_align])
	
	submit_button2.click(transcribe_custom,
						inputs=[model_select2, audio_upload2, audio_record2, device_select2, batch_size_slider2, compute_type_select2, language_select2, chunk_size_slider2, beam_size_slider2, release_memory_checkbox2, save_root2, save_audio2, save_transcription2, save_alignments2, save_in_subfolder2, preserve_name2, alignments_format2],
						outputs=[transcription_output2, alignments_output2, time_transcribe2, time_align2])
	
	release_memory_button.click(release_memory_models)
	release_memory_button2.click(release_memory_models)
	

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Transcribe audio files using WhisperX")
	parser.add_argument("--autolaunch", action="store_true", default=False, help="Launch the interface automatically in the default browser")
	parser.add_argument("--share", action="store_true", default=False, help="Create a share link to access the interface from another device")
	args = parser.parse_args()

	# Launch the interface
	print("Creating interface...")
	demo.launch(inbrowser=args.autolaunch, share=args.share)
