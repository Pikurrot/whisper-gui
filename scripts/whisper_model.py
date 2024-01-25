import torch
import whisperx
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
from typing import List

SAMPLE_RATE = 16000

class CustomWhisper():
	def __init__(
			self,
			model,
			processor,
			vad,
			vad_params,
			device,
			compute_type
	):
		self.model = model
		self.processor = processor
		self.vad = vad
		self.vad_params = vad_params
		self.device = device
		self.compute_type = compute_type
		self.model.to(device)
		
	def _transcribe_segments(self, audio_segments, language):
		transcriptions = []
		for audio in audio_segments:
			input_features = self.processor(
				audio,
				sampling_rate=SAMPLE_RATE,
				return_tensors="pt"
			).input_features.to(self.device).to(self.compute_type)

			predicted_ids = self.model.generate(input_features, language=language)

			transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
			transcriptions.append(transcription)
		return transcriptions
	
	def transcribe(
			self,
			audio,
			batch_size,
			language,
			chunk_size,
			print_progress
	):
		print("Obtaining VAD segments...")
		vad_segments = self.vad({
			"waveform": torch.from_numpy(audio).unsqueeze(0),
			"sample_rate": SAMPLE_RATE})
		print("VAD segments obtained.")
		if chunk_size is None:
			chunk_size = self.vad_params["chunk_size"]
		vad_segments = whisperx.vad.merge_chunks(
			vad_segments,
			chunk_size,
			onset=self.vad_params["vad_onset"],
			offset=self.vad_params["vad_offset"],
		)
		print("VAD segments merged.")
		print(f"Transcribing (language = {language})...")
		segments = []
		total_segments = len(vad_segments)
		for idx, out in enumerate(self._transcribe_segments(_audio_segment_gen(audio, vad_segments), language)):
			if print_progress:
				percent_complete = ((idx + 1) / total_segments) * 100
				print(f"Progress: {percent_complete:.2f}%...")
			text = out[0]
			segments.append(
				{
					"text": text,
					"start": round(vad_segments[idx]["start"], 3),
					"end": round(vad_segments[idx]["end"], 3)
				}
			)
		return {"segments": segments, "language": language}


def load_custom_model(
		model_id,
		device,
		compute_type,
		download_root,
		vad_model=None,
		vad_options=None,
):
	if isinstance(compute_type, str):
		if compute_type == "float32":
			compute_type = torch.float32
		elif compute_type == "float16":
			compute_type = torch.float16
		elif compute_type == "int8":
			compute_type = torch.int8
		else:
			raise ValueError(f"Unsupported compute_type: {compute_type}")

	is_local = _check_is_local(model_id, download_root)

	# Load Whisper model and processor
	processor = WhisperProcessor.from_pretrained(
		model_id, cache_dir=download_root, local_files_only=is_local
	)
	model = WhisperForConditionalGeneration.from_pretrained(
		model_id, torch_dtype=compute_type, use_safetensors=True, cache_dir=download_root, local_files_only=is_local
	)

	# Load VAD model
	default_vad_options = {
		"vad_onset": 0.500,
		"vad_offset": 0.363,
	}
	if vad_options is not None:
		default_vad_options.update(vad_options)
	if vad_model is None:
		print("Loading VAD model...")
		vad_model = whisperx.vad.load_vad_model(
			torch.device(device), 
			use_auth_token=None, 
			**default_vad_options
		)
		print("VAD model loaded.")
	default_vad_options["chunk_size"] = 16

	return CustomWhisper(model, processor, vad_model, default_vad_options, device, compute_type)

def _check_is_local(model_id, models_dir):
	model_folder = str(model_id).replace("/", "--")
	model_folder = "models--" + model_folder
	model_path = os.path.join(models_dir, model_folder)
	return os.path.exists(model_path)

def _audio_segment_gen(audio, segments):
	for seg in segments:
		f1 = int(seg["start"] * whisperx.audio.SAMPLE_RATE)
		f2 = int(seg["end"] * whisperx.audio.SAMPLE_RATE)
		yield audio[f1:f2]
