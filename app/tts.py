import sys
import os
import logging

# Add tacotron paths
sys.path.append('./tacotron_inference/hifigan')
sys.path.append('./tacotron_inference/tacotron2')

from tacotron_inference.inference import TextToSpeech

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTSService")

class TTSService:
    def __init__(self):
        """Initialize the TTS service with Tacotron2 + HiFi-GAN"""
        logger.info("Initializing Tacotron2 TTS Service...")
        self.tts = TextToSpeech()
        self._current_speaker = None

        # Map (lang, voice_model) to speaker names
        self.speaker_map = {
            ("en", "pecs_child"): "eng_chd",
            ("en", "pecs_man"): "eng_man",
            ("en", "pecs_woman"): "eng_wmn",
            ("kz", "pecs_child"): "kaz_chd",
            ("kz", "pecs_man"): "kaz_man",
            ("kz", "pecs_woman"): "kaz_wmn",
            ("ru", "pecs_child"): "rus_chd",
            ("ru", "pecs_man"): "rus_man",
            ("ru", "pecs_woman"): "rus_wmn",
        }

    def _ensure_speaker_loaded(self, lang: str, voice_model: str):
        """Ensure the correct speaker model is loaded for the language and voice_model"""
        speaker = self.speaker_map.get((lang, voice_model))
        if not speaker:
            logger.error(f"Unsupported language/model combination: {lang}/{voice_model}")
            raise ValueError(f"Unsupported language/model combination: {lang}/{voice_model}")
        if self._current_speaker != speaker:
            logger.info(f"Loading speaker for language '{lang}', model '{voice_model}': {speaker}")
            self.tts.load_speaker(speaker)
            self._current_speaker = speaker

    def synthesize(self, text: str, lang: str = "en", voice_model: str = "pecs_child") -> bytes:
        """
        Synthesize speech from text using the appropriate language model.
        Returns WAV file bytes.
        """
        logger.info(f"Synthesizing text for lang='{lang}', voice_model='{voice_model}': {text[:60]}{'...' if len(text) > 60 else ''}")
        # Ensure output directory exists
        os.makedirs('outputs', exist_ok=True)
        
        # Generate unique output filename
        output_file = f"outputs/temp_{hash(text + lang + voice_model)}.wav"
        
        try:
            # Load correct speaker for language and voice model
            self._ensure_speaker_loaded(lang, voice_model)
            
            # Synthesize audio
            logger.debug(f"Calling tts.synthesize with output_file={output_file}")
            self.tts.synthesize(text, output_file)
            
            # Read the generated file
            with open(output_file, 'rb') as f:
                audio_bytes = f.read()
            
            # Cleanup
            os.remove(output_file)
            
            logger.info(f"Synthesis complete for lang='{lang}', voice_model='{voice_model}'. Output file removed after reading.")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Speech synthesis failed: {str(e)}")