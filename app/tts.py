import sys
import os

# Add tacotron paths
sys.path.append('./tacotron_inference/hifigan')
sys.path.append('./tacotron_inference/tacotron2')

from tacotron_inference.inference import TextToSpeech

class TTSService:
    def __init__(self):
        """Initialize the TTS service with Tacotron2 + HiFi-GAN"""
        print("Initializing Tacotron2 TTS Service...")
        self.tts = TextToSpeech()
        self._current_speaker = None
        
        # Map language codes to default speakers
        self.language_speakers = {
            "en": "eng_wmn",
            "kz": "kaz_wmn", 
            "ru": "rus_wmn"
        }

    def _ensure_speaker_loaded(self, lang: str):
        """Ensure the correct speaker model is loaded for the language"""
        speaker = self.language_speakers.get(lang)
        if not speaker:
            raise ValueError(f"Unsupported language: {lang}")
        
        if self._current_speaker != speaker:
            self.tts.load_speaker(speaker)
            self._current_speaker = speaker

    def synthesize(self, text: str, lang: str = "en") -> bytes:
        """
        Synthesize speech from text using the appropriate language model.
        Returns WAV file bytes.
        """
        # Ensure output directory exists
        os.makedirs('outputs', exist_ok=True)
        
        # Generate unique output filename
        output_file = f"outputs/temp_{hash(text + lang)}.wav"
        
        try:
            # Load correct speaker for language
            self._ensure_speaker_loaded(lang)
            
            # Synthesize audio
            self.tts.synthesize(text, output_file)
            
            # Read the generated file
            with open(output_file, 'rb') as f:
                audio_bytes = f.read()
            
            # Cleanup
            os.remove(output_file)
            
            return audio_bytes
            
        except Exception as e:
            raise RuntimeError(f"Speech synthesis failed: {str(e)}")