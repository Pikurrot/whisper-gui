import torch
import whisperx
from whisperx.vad import VoiceActivitySegmentation
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
from typing import List, Optional, Collection, Dict, Any, Union
import numpy as np

SAMPLE_RATE = 16000

LANG_CODES = {"english": "en", "spanish": "es", "french": "fr", "german": "de", "italian": "it", "catalan": "ca", "chinese": "zh", "japanese": "ja", "portuguese": "pt", "arabic": "ar", "afrikaans": "af", "albanian": "sq", "amharic": "am", "armenian": "hy", "assamese": "as", "azerbaijani": "az", "bashkir": "ba", "basque": "eu", "belarusian": "be", "bengali": "bn", "bosnian": "bs", "breton": "br", "bulgarian": "bg", "burmese": "my", "castilian": "es", "croatian": "hr", "czech": "cs", "danish": "da", "dutch": "nl", "estonian": "et", "faroese": "fo", "finnish": "fi", "flemish": "nl", "galician": "gl", "georgian": "ka", "greek": "el", "gujarati": "gu", "haitian": "ht", "haitian creole": "ht", "hausa": "ha", "hebrew": "he", "hindi": "hi", "hungarian": "hu", "icelandic": "is", "indonesian": "id", "javanese": "jv", "kannada": "kn", "kazakh": "kk", "korean": "ko", "lao": "lo", "latin": "la", "latvian": "lv", "letzeburgesch": "lb", "lingala": "ln", "lithuanian": "lt", "luxembourgish": "lb", "macedonian": "mk", "malagasy": "mg", "malay": "ms", "malayalam": "ml", "maltese": "mt", "maori": "mi", "marathi": "mr", "moldavian": "ro", "moldovan": "ro", "mongolian": "mn", "nepali": "ne", "norwegian": "no", "occitan": "oc", "panjabi": "pa", "pashto": "ps", "persian": "fa", "polish": "pl", "punjabi": "pa", "pushto": "ps", "romanian": "ro", "russian": "ru", "sanskrit": "sa", "serbian": "sr", "shona": "sn", "sindhi": "sd", "sinhala": "si", "sinhalese": "si", "slovak": "sk", "slovenian": "sl", "somali": "so", "sundanese": "su", "swahili": "sw", "swedish": "sv", "tagalog": "tl", "tajik": "tg", "tamil": "ta", "tatar": "tt", "telugu": "te", "thai": "th", "tibetan": "bo", "turkish": "tr", "turkmen": "tk", "ukrainian": "uk", "urdu": "ur", "uzbek": "uz", "valencian": "ca", "vietnamese": "vi", "welsh": "cy", "yiddish": "yi", "yoruba": "yo"}

class CustomWhisper():
	def __init__(
			self,
			model: WhisperForConditionalGeneration,
			processor: WhisperProcessor,
			vad: VoiceActivitySegmentation,
			vad_params: Dict[str, Any],
			device: str,
			compute_type: torch.dtype
	):
		"""
		Custom Whisper model. Takes any valid whisper model, its processor and a VAD model and allows transcribing audio.

		Recommended to instantiate with load_custom_model().
		"""
		self.model = model
		self.processor = processor
		self.vad = vad
		self.vad_params = vad_params
		self.device = device
		self.compute_type = compute_type
		self.model.to(device)
		
	def _transcribe_segments(self, audio_batches, language):
		transcriptions = []
		for audio_batch in audio_batches:
			input_features = self.processor(
				audio_batch,
				sampling_rate=SAMPLE_RATE,
				return_tensors="pt",
				padding=True  # Ensure all sequences in the batch have the same length
			).input_features.to(self.device).to(self.compute_type)

			required_length = 3000
			current_length = input_features.shape[2]
			if current_length < required_length:
				padding_length = required_length - current_length
				padding = torch.zeros((input_features.shape[0], input_features.shape[1], padding_length), device=self.device, dtype=self.compute_type)
				input_features = torch.cat([input_features, padding], dim=2)

			predicted_ids = self.model.generate(input_features, language=language)

			batch_transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
			transcriptions.extend(batch_transcriptions)
		return transcriptions
	
	def transcribe(
			self,
			audio: np.ndarray,
			batch_size: int = 1,
			language: str = None,
			chunk_size: int = 20,
			print_progress: bool = True
	) -> Dict[str, List[Dict[str, Any]]]:
		"""
		Transcribe a given audio array.
		
		Returns as:
			{"segments": [
				{
					"text": str,
					"start": float,
					"end": float
				}, ...],
			 "language": str
			 }
		"""
		# Obtain VAD segments (timestamps where speech is detected)
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

		# Obtain language
		lang_code = LANG_CODES.get(language, None)
		if lang_code is None:
			# Detect language for first 30s of audio (max duration a Whisper model allows)
			print("Language not specified. Detecting language...")
			input_features = self.processor(
				audio,
				sampling_rate=SAMPLE_RATE,
				return_tensors="pt"
			).input_features.to(self.device).to(self.compute_type)
			print(type(input_features))
				
			language_tokens = [t[2:-2] for t in self.processor.tokenizer.additional_special_tokens if len(t) == 6]
			possible_languages = list(set(language_tokens).intersection(LANG_CODES.values()))
			lang_code = self._detect_language(input_features, possible_languages)[0]
			language = list(LANG_CODES.keys())[list(LANG_CODES.values()).index(lang_code)]
			print(f"Language detected in the first 30s: {language}")

		# Transcribe
		print(f"Transcribing (language = {language})...")
		segments = _audio_segment_gen(audio, vad_segments)
		audio_batches = []
		current_batch = []
		# prepare batches of size batch_size for batch inference
		for segment in segments:
			current_batch.append(segment)
			if len(current_batch) == batch_size:
				audio_batches.append(current_batch)
				current_batch = []
		if current_batch:  # Add the last batch (can have different size) if it has any segments
			audio_batches.append(current_batch)

		final_transcriptions = []
		total_batches = len(audio_batches)
		for idx, audio_batch in enumerate(audio_batches):
			if print_progress:
				percent_complete = ((idx + 1) / total_batches) * 100
				print(f"Processing batch {idx + 1}/{total_batches}, Progress: {percent_complete:.2f}%...")

			batch_transcriptions = self._transcribe_segments(audio_batch, lang_code)
			
			for segment_idx, transcription in enumerate(batch_transcriptions):
				actual_segment_idx = idx * batch_size + segment_idx
				if actual_segment_idx < len(vad_segments):
					vad_segment = vad_segments[actual_segment_idx]
					final_transcriptions.append({
						"text": transcription,
						"start": round(vad_segment["start"], 3),
						"end": round(vad_segment["end"], 3)
					})

		return {"segments": final_transcriptions, "language": lang_code}

	def _detect_language(
			self,
			input_features: torch.Tensor,
			possible_languages: Optional[Collection[str]] = None
	) -> List[Dict[str, float]]:
		# hacky, but all language tokens and only language tokens are 6 characters long
		language_tokens = [t for t in self.processor.tokenizer.additional_special_tokens if len(t) == 6]
		if possible_languages is not None:
			language_tokens = [t for t in language_tokens if t[2:-2] in possible_languages]
			if len(language_tokens) < len(possible_languages):
				raise RuntimeError(f'Some languages in {possible_languages} did not have associated language tokens')

		language_token_ids = self.processor.tokenizer.convert_tokens_to_ids(language_tokens)

		# 50258 is the token for transcribing
		logits = self.model(input_features,
						decoder_input_ids = torch.tensor([[50258] for _ in range(input_features.shape[0])], device=self.device)).logits
		mask = torch.ones(logits.shape[-1], dtype=torch.bool)
		mask[language_token_ids] = False
		logits[:, :, mask] = -float('inf')

		output_probs = logits.softmax(dim=-1).cpu()
		lang_probs= [
			{
				lang: output_probs[input_idx, 0, token_id].item()
				for token_id, lang in zip(language_token_ids, language_tokens)
			}
			for input_idx in range(logits.shape[0])
		]
		return [max(prob_dict, key=prob_dict.get)[2:-2] for prob_dict in lang_probs]

def load_custom_model(
		model_id: str,
		device: Union[str, torch.device],
		compute_type: str = "float32",
		download_root: str = "models/custom",
		vad_model: Optional[VoiceActivitySegmentation] = None,
		vad_options: Optional[Dict[str, Any]] = None
):
	"""
	Load a custom Whisper model local or, if not detected, download it from HuggingFace. Returns an instance of a CustomWhisper model.
	"""
	if isinstance(compute_type, str):
		if compute_type == "float32":
			compute_type = torch.float32
		elif compute_type == "float16":
			compute_type = torch.float16
		else:
			raise ValueError(f"Unsupported compute_type: {compute_type}")

	# Check if model is already downloaded
	is_local = _check_is_local(model_id, download_root)

	# Load Whisper model and processor
	processor = WhisperProcessor.from_pretrained(
		model_id, cache_dir=download_root, local_files_only=is_local
	)
	model = WhisperForConditionalGeneration.from_pretrained(
		model_id, torch_dtype=compute_type, use_safetensors=True, cache_dir=download_root, local_files_only=is_local
	)
	print("Model loaded.")

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
