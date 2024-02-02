import gradio as gr
import whisperx
import subprocess, os, gc, argparse, shutil, inspect
import soundfile as sf
import torch, re, json, time
from datetime import datetime
from scripts.whisper_model import load_custom_model, LANG_CODES
from typing import Optional, Tuple, Callable
from scripts.config_io import read_config_value

ALIGN_LANGS = ["en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl", "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca", "ml", "no", "nn"]

# global variables
g_model = None
g_model_a = None
g_model_a_metadata = None
g_params = {}

def list_models():
	models_dir = os.path.join("models", "custom")
	subdirs = os.listdir(models_dir)
	models = [s for s in subdirs if s.startswith("models--")]
	models = [s.replace("models--", "").replace("--", "/") for s in models]
	return models

def create_save_folder(save_root):
	# Get the current date and create the date folder
	current_date = datetime.now().strftime("%Y-%m-%d")
	date_dir = os.path.join(save_root, current_date)
	
	# Create the directory if it doesn"t exist
	if not os.path.exists(date_dir):
		os.makedirs(date_dir)

	# Determine the highest counter already used in folder names
	highest_counter = -1
	for existing_folder in os.listdir(date_dir):
		match = re.match("(\d+)", existing_folder)
		if match:
			highest_counter = max(highest_counter, int(match.group(1)))
	
	# Determine the next available folder with counter
	next_counter = highest_counter + 1
	save_dir = os.path.join(date_dir, f"{next_counter:04d}")  # Format as four digits

	# Create the counter folder
	os.makedirs(save_dir)
	return save_dir

def save_audio_to_mp3(audio_tuple, save_dir):
	rate, y = audio_tuple
	if len(y.shape) == 2:
		y = y.T[0]  # If stereo, take one channel
	audio_path = os.path.join(save_dir, "audio.mp3")
	wav_path = os.path.join("temp", "temp_audio.wav")
	sf.write(wav_path, y, rate)
	subprocess.run(["ffmpeg", "-i", wav_path, audio_path])  # Convert WAV to MP3
	os.remove(wav_path)  # Remove temporary WAV file
	return audio_path

def save_transcription_to_txt(text_str, save_dir):
	text_path = os.path.join(save_dir, "transcription.txt")
	with open(text_path, "w", encoding="utf-8") as f:
		f.write(text_str)
	return text_path

def save_alignments_to_json(alignment_dict, save_dir):
	json_path = os.path.join(save_dir, "alignments.json")
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(alignment_dict, f, indent=4)
	return json_path

def load_and_save_audio(audio_path, micro_audio, save_audio, save_dir):
	if micro_audio:
		print("Saving micro audio...")
		audio_path = save_audio_to_mp3(micro_audio, save_dir if save_audio else "temp")
	elif save_audio:
		print("Making a copy of the audio...")
		shutil.copy(audio_path, os.path.join(save_dir, "audio.mp3"))

	print("Loading audio...")
	audio = whisperx.load_audio(audio_path)
	
	if micro_audio and not save_audio:
		os.remove(audio_path)
	
	return audio

def float_to_time_str(time_float):
	seconds_total = int(time_float)
	hours = seconds_total // 3600
	minutes = (seconds_total % 3600) // 60
	seconds = seconds_total % 60
	if hours > 0:
		time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
	else:
		time_str = f"{minutes:02d}:{seconds:02d}"
	return time_str

def format_alignments(alignments):
	formatted_transcription = []
	for segment in alignments["segments"]:
		start_time = float_to_time_str(segment["start"])
		end_time = float_to_time_str(segment["end"])
		text = segment["text"].strip()
		# Format the line as "start_time - end_time: text"
		formatted_line = f"{start_time} - {end_time}: {text}"
		formatted_transcription.append(formatted_line)
	return "\n\n".join(formatted_transcription)

def release_whisper():
	global g_model, g_params
	del g_model
	if g_params.get("device", None) == "cuda": torch.cuda.empty_cache()
	else: gc.collect()
	g_model = None
	print("Whisper model released from memory")

def release_align():
	global g_model_a, g_params
	del g_model_a
	if g_params.get("device", None) == "cuda": torch.cuda.empty_cache()
	else: gc.collect()
	g_model_a = None
	print("Alignment model released from memory")

def release_memory_models():
	global g_model, g_model_a, g_params
	del g_model, g_model_a
	if g_params.get("device", None) == "cuda": torch.cuda.empty_cache()
	else: gc.collect()
	g_model = None
	g_model_a = None
	print("Models released from memory")

def get_args_str(func: Callable) -> list:
	return list(inspect.signature(func).parameters)

def get_params(func: Callable, values: list) -> dict:
	keys = get_args_str(func)
	return {k: values[k] for k in keys}

def same_params(params1: dict, params2: dict, *args):
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
		save_root: Optional[bool],
		save_audio: bool,
		save_transcription: bool,
		save_alignments: bool
	) -> Tuple[str, str, str, str]:

	print("Inputs received. Starting...")
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
		g_model = whisperx.load_model(model_name, device, compute_type=compute_type, asr_options={"beam_size": beam_size}, download_root="models/whisperx")
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
		save_root: Optional[bool],
		save_audio: bool,
		save_transcription: bool,
		save_alignments: bool
	) -> Tuple[str, str, str, str]:

	print("Inputs received. Starting...")
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
		g_model = load_custom_model(model_name, device, compute_type=compute_type, beam_size=beam_size, download_root="models/custom")
	g_params = params

	return _transcribe()

def _transcribe() -> Tuple[str, str]:
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
		save_dir = create_save_folder(save_root)

	# Load (and save) audio
		print("Loading audio...")
	audio = load_and_save_audio(g_params["audio_path"], g_params["micro_audio"], g_params["save_audio"], save_dir)

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
		save_transcription_to_txt(joined_text, save_dir)

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
		save_alignments_to_json(aligned_result, save_dir)
	if g_params["release_memory"]:
		release_align()
	print("Done!")
	if not os.listdir("temp"):
		# Remove temp folder if empty
		os.rmdir("temp")
	# Return the transcription and sentence-level alignments
	return joined_text, format_alignments(aligned_result), f"{round(time_transcribe, 3)}s", f"{round(time_align, 3)}s"


def main():
	# Parse arguments
	parser = argparse.ArgumentParser(description="Transcribe audio files using WhisperX")
	parser.add_argument("--autolaunch", action="store_true", default=False, help="Launch the interface automatically in the default browser")
	parser.add_argument("--share", action="store_true", default=False, help="Create a share link to access the interface from another device")
	args = parser.parse_args()

	# Prepare interface data
	whisperx_models = ["large-v2", "large-v1", "large", "medium", "small", "base", "tiny", "medium.en", "small.en", "base.en", "tiny.en"]
	custom_models = list_models()
	whisperx_langs = ["auto", "en", "es", "fr", "de", "it", "ja", "zh", "nl", "uk", "pt"]
	custom_langs = ["auto"] + list(LANG_CODES.keys())
	release_whisper_message = "When changed, requires the Whisper model to reload."
	release_align_message = "When changed, requires the alignment model to reload."
	release_both_message = "When changed, requires both models to reload."

	# Read config
	gpu_support, error = read_config_value("gpu_support")
	if gpu_support:
		device = "cuda"
		device_interactive = True
		device_message = ""
	else:
		device = "cpu"
		device_interactive = False
		if gpu_support is None:
			device_message = "If you don\"t have a GPU, select \"cpu\".\n"
		else:
			device_message = "GPU support is disabled in the config file.\n"

	# Create Gradio Interface
	print("Creating interface...")
	with gr.Blocks(title="Whisper GUI") as gui:
		gr.Markdown("""# Whisper GUI
A simple interface to transcribe audio files using the Whisper model""")
		with gr.Tab("WhisperX"):
			with gr.Row():
				with gr.Column():
					model_select = gr.Dropdown(whisperx_models, value="base", label="Load WhisperX Model", info=release_whisper_message)
					with gr.Group():
						audio_upload = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio File")
						audio_record = gr.Audio(sources=["microphone"], type="numpy", label="or Record Audio (If both are provided, only microphone audio will be used)")
						save_audio = gr.Checkbox(value=False, label="Save Audio")
					gr.Examples(examples=["examples/coffe_break_example.mp3"], inputs=audio_upload)
					with gr.Accordion(label="Advanced Options", open=False):
						language_select = gr.Dropdown(whisperx_langs, value = "auto", label="Language", info="Select the language of the audio file. Select \"auto\" to automatically detect it. "+release_align_message)
						device_select = gr.Radio(["cuda", "cpu"], value = device, label="Device", info=device_message+release_both_message, interactive=device_interactive)
						with gr.Group():
							with gr.Row():
								save_transcription = gr.Checkbox(value=True, label="Save Transcription")
								save_alignments = gr.Checkbox(value=True, label="Save Alignments")
							save_root = gr.Textbox(label="Save Path", placeholder="outputs", lines=1)
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
						audio_upload2 = gr.Audio(sources=["upload"], type="filepath", label="Load Audio File")
						audio_record2 = gr.Audio(sources=["microphone"], type="numpy", label="or Record Audio (If both are provided, only microphone audio will be used)")
						save_audio2 = gr.Checkbox(value=False, label="Save Audio")
					gr.Examples(examples=["examples/coffe_break_example.mp3"], inputs=audio_upload2)
					with gr.Accordion(label="Advanced Options", open=False):
						language_select2 = gr.Dropdown(custom_langs, value = "auto", label="Language", info="Select the language of the audio file. Select \"auto\" to automatically detect it. "+release_align_message)
						device_select2 = gr.Radio(["cuda", "cpu"], value = device, label="Device", info=device_message+release_both_message, interactive=device_interactive)
						with gr.Group():
							with gr.Row():
								save_transcription2 = gr.Checkbox(value=True, label="Save Transcription")
								save_alignments2 = gr.Checkbox(value=True, label="Save Alignments")
							save_root2 = gr.Textbox(label="Save Path", placeholder="outputs", lines=1)
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
					  		inputs=[model_select, audio_upload, audio_record, device_select, batch_size_slider, compute_type_select, language_select, chunk_size_slider, beam_size_slider, release_memory_checkbox, save_root, save_audio, save_transcription, save_alignments],
							outputs=[transcription_output, alignments_output, time_transcribe, time_align])
		
		submit_button2.click(transcribe_custom,
					  		inputs=[model_select2, audio_upload2, audio_record2, device_select2, batch_size_slider2, compute_type_select2, language_select2, chunk_size_slider2, beam_size_slider2, release_memory_checkbox2, save_root2, save_audio2, save_transcription2, save_alignments2],
							outputs=[transcription_output2, alignments_output2, time_transcribe2, time_align2])
		
		release_memory_button.click(release_memory_models)
		release_memory_button2.click(release_memory_models)

	# Launch the interface
	gui.launch(inbrowser=args.autolaunch, share=args.share)

if __name__ == "__main__":
	main()
