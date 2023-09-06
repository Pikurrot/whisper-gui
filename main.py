import gradio as gr
import whisperx
import subprocess, os, gc, argparse
import soundfile as sf
import torch, re

def save_audio_to_mp3(audio_tuple, save_dir='recordings', base_filename='recording'):
	rate, y = audio_tuple
	if len(y.shape) == 2:
		y = y.T[0]  # If stereo, take one channel
	
	# Create the directory if it doesn't exist
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	# Check for base filename without counter
	base_exists = os.path.exists(os.path.join(save_dir, f"{base_filename}.mp3"))
	# Determine the highest counter already used in filenames
	highest_counter = 0
	for existing_file in os.listdir(save_dir):
		match = re.match(f"{base_filename}_(\d+)\.mp3", existing_file)
		if match:
			highest_counter = max(highest_counter, int(match.group(1)))
	# Determine the next available save_path
	next_counter = highest_counter + 1
	if not base_exists:
		save_path = os.path.join(save_dir, f"{base_filename}.mp3")
	else:
		save_path = os.path.join(save_dir, f"{base_filename}_{next_counter}.mp3")

	# Save as WAV file first
	wav_path = os.path.join(save_dir, 'temp_audio.wav')
	sf.write(wav_path, y, rate)

	# Convert WAV to MP3 using ffmpeg
	subprocess.run(['ffmpeg', '-i', wav_path, save_path])

	# Remove the temporary WAV file
	os.remove(wav_path)

	return save_path

def transcribe_audio(model_name, audio_path, micro_audio, device, batch_size, compute_type, language, chunk_size, release_memory):
	# Transcription
	print('Loading model...')
	model = whisperx.load_model(model_name, device, compute_type=compute_type, download_root='models')
	print('Loading audio...')
	if micro_audio:
		audio_path = save_audio_to_mp3(micro_audio)
	audio = whisperx.load_audio(audio_path)
	print('Transcribing...')
	if language == 'auto': language = None
	result = model.transcribe(audio, batch_size=batch_size, language=language, chunk_size=chunk_size, print_progress=True)
	if release_memory:
		del model
		if device == 'cuda': torch.cuda.empty_cache()
		else: gc.collect()

	# Alignment
	print('Loading alignment model...')
	model_a, metadata = whisperx.load_align_model(language_code=result['language'], device=device, model_dir='models/alignment')
	print('Aligning...')
	aligned_result = whisperx.align(result['segments'], model_a, metadata, audio, device, return_char_alignments=False)
	if release_memory:
		del model_a, metadata
		if device == 'cuda': torch.cuda.empty_cache()
		else: gc.collect()
	print('Done!')
	return ' '.join([segment['text'] for segment in aligned_result['segments']])

def main():
	# Parse arguments
	parser = argparse.ArgumentParser(description='Transcribe audio files using WhisperX')
	parser.add_argument('--autolaunch', action='store_true', default=False, help='Launch the interface automatically in the default browser')
	parser.add_argument('--share', action='store_true', default=False, help='Create a share link to access the interface from another device')
	args = parser.parse_args()

	# Create Gradio Interface
	print('Creating interface...')
	with gr.Blocks(title='Whisper GUI') as gui:
		gr.Markdown('''# Whisper GUI
A simple interface to transcribe audio files using the Whisper model''')
		with gr.Row():
			with gr.Column():
				model_select = gr.Dropdown(['large-v2', 'large-v1', 'large', 'medium', 'small', 'base', 'tiny', 'medium.en', 'small.en', 'base.en', 'tiny.en'], value='large-v2', label='Load Model')
				with gr.Row():
					audio_upload = gr.Audio(source='upload', type='filepath', label='Upload Audio File')
					audio_record = gr.Audio(source='microphone', type='numpy', label='or Record Audio (If both are provided, only microphone audio will be used)')
				with gr.Accordion(label='Advanced Options', open=False):
					language_select = gr.Dropdown(['auto', 'en', 'es', 'fr', 'de', 'it', 'ja', 'zh', 'nl', 'uk', 'pt'], value = 'auto', label='Language', info='Select the language of the audio file. Select "auto" to automatically detect it.')
					device_select = gr.Radio(['cuda', 'cpu'], value = 'cuda', label='Device', info='If you don\'t have a GPU, select "cpu"')
					gr.Markdown('''### Optimizations''')
					compute_type_select = gr.Radio(['int8', 'float16', 'float32'], value = 'int8', label='Compute Type', info='int8 is fastest and requires less memory. float32 is more accurate (The model or your device may not support some data types)')
					batch_size_slider = gr.Slider(1, 128, value = 1, label='Batch Size', info='Larger batch sizes may be faster but require more memory')
					chunk_size_slider = gr.Slider(1, 80, value = 20, label='Chunk Size', info='Larger chunk sizes may be faster but require more memory')
					release_memory_checkbox = gr.Checkbox(label='Release Memory', value=True, info='Release model from memory after every transcription')
				submit_button = gr.Button(value='Start Transcription')
			transcription_output = gr.Textbox(label='Transcription')
		
		submit_button.click(transcribe_audio,
					  		inputs=[model_select, audio_upload, audio_record, device_select, batch_size_slider, compute_type_select, language_select, chunk_size_slider, release_memory_checkbox],
							outputs=transcription_output)

	# Launch the interface
	gui.launch(inbrowser=args.autolaunch, share=args.share)

if __name__ == '__main__':
	main()
