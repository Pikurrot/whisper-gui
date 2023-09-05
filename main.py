import gradio as gr
import whisperx
import subprocess, os
import soundfile as sf

def save_audio_to_mp3(audio_tuple, save_path='audios/audio.mp3'):
    rate, y = audio_tuple
    if len(y.shape) == 2:
        y = y.T[0]  # If stereo, take one channel

    # Save as WAV file first
    wav_path = 'audios/temp_audio.wav'
    sf.write(wav_path, y, rate)

	# Convert WAV to MP3 using ffmpeg
    subprocess.run(['ffmpeg', '-i', wav_path, save_path])

    # Remove the temporary WAV file
    os.remove(wav_path)

def transcribe_audio(audio_tuple):
	device = 'cuda'  # or 'cpu'
	batch_size = 1
	compute_type = 'float32'

	# save copy of audio
	save_audio_to_mp3(audio_tuple)

	# Transcription
	print('Loading model...')
	model = whisperx.load_model('large-v2', device, compute_type=compute_type, download_root='models')
	print('Loading audio...')
	audio = whisperx.load_audio('audio.mp3')
	print('Transcribing...')
	result = model.transcribe(audio, batch_size=batch_size, language='es')

	# Alignment
	print('Loading alignment model...')
	model_a, metadata = whisperx.load_align_model(language_code=result['language'], device=device, model_dir='models/alignment')
	print('Aligning...')
	aligned_result = whisperx.align(result['segments'], model_a, metadata, audio, device, return_char_alignments=False)
	print('Done!')
	return ' '.join([segment['text'] for segment in aligned_result['segments']])

def main():
	# Create Gradio Interface
	iface = gr.Interface(
		transcribe_audio,
		gr.Audio(source='upload', label='Upload Audio File'),
		gr.outputs.Textbox(label='Transcription'),
		allow_flagging=False,
	)

	# Launch the interface
	iface.launch()

if __name__ == '__main__':
	main()
