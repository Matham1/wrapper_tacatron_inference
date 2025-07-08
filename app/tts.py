from dotenv import load_dotenv
load_dotenv()

import onnxruntime
import utils
# We assume this function can be modified or we get the audio data from it
from infer_onnx import synthesize_speech_to_memory 
import io

class TTSService:
    def __init__(self, model_path: str, config_path: str, lang: str = "en", sid: int = None, use_accent: bool = True):
        self.model_path = model_path
        self.config_path = config_path
        self.lang = lang
        self.sid = sid
        self.use_accent = use_accent

        sess_options = onnxruntime.SessionOptions()
        self.model = onnxruntime.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.hps = utils.get_hparams_from_file(self.config_path)

    def synthesize(self, text: str) -> bytes:
        """
        Synthesizes speech and returns the audio data as bytes from an in-memory buffer.
        """
        # This new function returns raw audio data and sampling rate
        audio, sampling_rate = synthesize_speech_to_memory(
            text, self.model, self.hps, self.lang, self.sid, self.use_accent
        )

        # Use soundfile to write the raw audio to an in-memory buffer
        buffer = io.BytesIO()
        import soundfile as sf
        sf.write(buffer, audio, sampling_rate, format='WAV')
        
        # Go to the beginning of the buffer to read its contents
        buffer.seek(0)
        return buffer.read()