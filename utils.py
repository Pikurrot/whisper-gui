import whisperx
import subprocess
import shutil
import soundfile as sf
import re
import os
import json
from datetime import datetime

def list_models():
	models_dir = os.path.join("models", "custom")
	if not os.path.exists(models_dir):
		# create the directory if it doesn"t exist
		os.makedirs(models_dir)
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

def save_transcription_to_txt(text_str, save_dir, name="transcription.txt"):
	text_path = os.path.join(save_dir, name)
	print(f"Saving transcription to {text_path}...")
	with open(text_path, "w", encoding="utf-8") as f:
		f.write(text_str)
	return text_path

def save_alignments_to_json(alignment_dict, save_dir, name="alignments.json"):
	json_path = os.path.join(save_dir, name)
	print(f"Saving alignments to {json_path}...")
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(alignment_dict, f, indent=4)
	return json_path

def save_alignments_to_srt(subtitles_dict, save_dir, name="subtitles.srt"):
	srt_path = os.path.join(save_dir, name)
	print(f"Saving subtitles to {srt_path}...")
	with open(srt_path, "w", encoding="utf-8") as f:
		for sub in subtitles_dict:
			f.write(f"{sub['number']}\n{sub['start']} --> {sub['end']}\n{sub['text']}\n\n")
	return srt_path

def load_and_save_audio(audio_path, micro_audio, save_audio, save_dir, preserve_name=False):
	if micro_audio:
		print("Saving micro audio...")
		audio_path = save_audio_to_mp3(micro_audio, save_dir if save_audio else "temp")
	elif save_audio:
		print("Making a copy of the audio...")
		original_name = os.path.basename(audio_path)
		shutil.copy(audio_path, os.path.join(save_dir, original_name if preserve_name else "audio.mp3"))

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

def alignments2subtitles(subtitles, max_line_length=40):
	def sec2timesrt(sec):
		# Convert seconds to HH:MM:SS,mmm format
		hours, remainder = divmod(sec, 3600)
		minutes, seconds = divmod(remainder, 60)
		milliseconds = int((seconds - int(seconds)) * 1000)
		return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

	def split_text(text, max_line_length):
		# Split text into multiple lines based on max line length
		lines = []
		while text:
			if len(text) <= max_line_length:
				lines.append(text)
				break
			else:
				# Find the nearest space before max_line_length
				split_at = text.rfind(" ", 0, max_line_length + 1)
				if split_at == -1:  # No spaces found, force split
					split_at = max_line_length
				lines.append(text[:split_at])
				text = text[split_at+1:]  # Skip the space
		return "\n".join(lines)

	converted_subtitles = []
	for index, sub in enumerate(subtitles, start=1):
		converted_sub = {
			"number": index,
			"start": sec2timesrt(sub["start"]),
			"end": sec2timesrt(sub["end"]),
			"text": split_text(sub["text"], max_line_length)
		}
		converted_subtitles.append(converted_sub)

	return converted_subtitles
