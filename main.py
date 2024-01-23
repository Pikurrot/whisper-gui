import gradio as gr
import whisperx
import subprocess, os, gc, argparse, shutil
import soundfile as sf
import torch, re, json
from datetime import datetime

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

def transcribe_audio(model_name, audio_path, micro_audio, device, batch_size, compute_type, language, chunk_size, release_memory, save_root, save_audio, save_transcription, save_alignments):
	print("Inputs received. Starting...")
	# Create save folder
	save_dir = None
	if not os.path.exists("temp"):
		os.makedirs("temp")
	if save_audio or save_transcription or save_alignments:
		if not save_root: save_root = "outputs"
		save_dir = create_save_folder(save_root)

	# Transcription
	print("Loading model...")
	model = whisperx.load_model(model_name, device, compute_type=compute_type, download_root="models")
	print("Loading audio...")
	audio = load_and_save_audio(audio_path, micro_audio, save_audio, save_dir)

	print("Transcribing...")
	if language == "auto": language = None
	result = model.transcribe(audio, batch_size=batch_size, language=language, chunk_size=chunk_size, print_progress=True)
	joined_text = " ".join([segment["text"].strip() for segment in result["segments"]])
	if save_transcription:
		save_transcription_to_txt(joined_text, save_dir)
	if release_memory:
		del model
		if device == "cuda": torch.cuda.empty_cache()
		else: gc.collect()

	# Alignment
	print("Loading alignment model...")
	model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, model_dir="models/alignment")
	print("Aligning...")
	aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
	if save_alignments:
		save_alignments_to_json(aligned_result, save_dir)
	if release_memory:
		del model_a, metadata
		if device == "cuda": torch.cuda.empty_cache()
		else: gc.collect()
	print("Done!")
	return joined_text, format_alignments(aligned_result)

def main():
	# Parse arguments
	parser = argparse.ArgumentParser(description="Transcribe audio files using WhisperX")
	parser.add_argument("--autolaunch", action="store_true", default=False, help="Launch the interface automatically in the default browser")
	parser.add_argument("--share", action="store_true", default=False, help="Create a share link to access the interface from another device")
	args = parser.parse_args()

	# Create Gradio Interface
	print("Creating interface...")
	with gr.Blocks(title="Whisper GUI") as gui:
		gr.Markdown("""# Whisper GUI
A simple interface to transcribe audio files using the Whisper model""")
		with gr.Row():
			with gr.Column():
				model_select = gr.Dropdown(["large-v2", "large-v1", "large", "medium", "small", "base", "tiny", "medium.en", "small.en", "base.en", "tiny.en"], value="base", label="Load Model")
				with gr.Row():
					audio_upload = gr.Audio(source="upload", type="filepath", label="Upload Audio File")
					audio_record = gr.Audio(source="microphone", type="numpy", label="or Record Audio (If both are provided, only microphone audio will be used)")
				gr.Examples(examples=["examples/coffe_break_example.mp3"], inputs=audio_upload)
				with gr.Accordion(label="Advanced Options", open=False):
					language_select = gr.Dropdown(["auto", "en", "es", "fr", "de", "it", "ja", "zh", "nl", "uk", "pt"], value = "auto", label="Language", info="Select the language of the audio file. Select \"auto\" to automatically detect it.")
					device_select = gr.Radio(["cuda", "cpu"], value = "cuda", label="Device", info="If you don\"t have a GPU, select \"cpu\"")
					with gr.Row():
						save_audio = gr.Checkbox(value=False, label="Save Audio")
						save_transcription = gr.Checkbox(value=True, label="Save Transcription")
						save_alignments = gr.Checkbox(value=True, label="Save Alignments")
					save_root = gr.Textbox(label="Save Path", placeholder="outputs", lines=1)
					gr.Markdown("""### Optimizations""")
					compute_type_select = gr.Radio(["int8", "float16", "float32"], value = "int8", label="Compute Type", info="int8 is fastest and requires less memory. float32 is more accurate (The model or your device may not support some data types)")
					batch_size_slider = gr.Slider(1, 128, value = 1, label="Batch Size", info="Larger batch sizes may be faster but require more memory")
					chunk_size_slider = gr.Slider(1, 80, value = 20, label="Chunk Size", info="Larger chunk sizes may be faster but require more memory")
					release_memory_checkbox = gr.Checkbox(label="Release Memory", value=True, info="Release model from memory after every transcription")
				submit_button = gr.Button(value="Start Transcription")
			with gr.Column():
				transcription_output = gr.Textbox(label="Transcription", lines=15)
				alignments_output = gr.Textbox(label="Alignments", lines=15)
		
		submit_button.click(transcribe_audio,
					  		inputs=[model_select, audio_upload, audio_record, device_select, batch_size_slider, compute_type_select, language_select, chunk_size_slider, release_memory_checkbox, save_root, save_audio, save_transcription, save_alignments],
							outputs=[transcription_output, alignments_output])

	# Launch the interface
	gui.launch(inbrowser=args.autolaunch, share=args.share)

if __name__ == "__main__":
	main()
