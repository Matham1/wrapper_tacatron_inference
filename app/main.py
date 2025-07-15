import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from app.tts import TTSService

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"

    @property
    def validated_lang(self) -> str:
        """Validate language code"""
        if self.lang not in ["en", "kz", "ru"]:
            raise ValueError(f"Unsupported language: {self.lang}")
        return self.lang

app = FastAPI()

# Global TTS service instance
tts_service: TTSService = None

@app.on_event("startup")
async def initialize_tts():
    """Initialize the TTS service on startup"""
    global tts_service
    try:
        tts_service = TTSService()
    except Exception as e:
        print(f"Failed to initialize TTS service: {e}")
        raise RuntimeError(f"TTS service initialization failed: {str(e)}")

@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    """Synthesize speech from text using Tacotron2"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    try:
        # Get validated language
        lang = req.validated_lang
        
        # Synthesize audio
        wav_bytes = tts_service.synthesize(req.text, lang)

        # Save a copy locally
        save_dir = "tts_audios"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        filename = f"tts_{req.lang}_{timestamp}.wav"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, "wb") as f:
            f.write(wav_bytes)
        print(f"[saved] {filepath}")

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f"inline; filename=tts_{req.lang}.wav"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Tacotron2 TTS"}