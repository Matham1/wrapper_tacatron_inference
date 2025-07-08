import io
import os
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel
from typing import Dict
from app.tts import TTSService

class TTSRequest(BaseModel):
    text: str
    lang: str = Query("en", enum=["en", "kz", "ru"])

app = FastAPI()

# A dictionary to hold different TTS service instances for each language
tts_services: Dict[str, TTSService] = {}

@app.on_event("startup")
async def load_models():
    """Load all available TTS models on application startup."""
    global tts_services
    # Define model paths and configurations for each language
    model_configs = {
        "en": {"model_path": "vits2_eng_girl.onnx", "config_path": "config_eng_girl.json"},
        "kz": {"model_path": "vits2_kaz_girl.onnx", "config_path": "config_kaz_girl.json"},
        "ru": {"model_path": "vits2_rus_girl.onnx", "config_path": "config_rus_girl.json"},
    }

    for lang, paths in model_configs.items():
        # Check if the files exist before loading
        if os.path.exists(paths["model_path"]) and os.path.exists(paths["config_path"]):
            tts_services[lang] = TTSService(
                model_path=paths["model_path"],
                config_path=paths["config_path"],
                lang=lang
            )
        else:
            # Handle cases where model files might be missing
            print(f"Warning: Model files for language '{lang}' not found. Skipping.")


@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    """Synthesize speech from text using the selected language model."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    # Get the appropriate TTS service based on the requested language
    tts_service = tts_services.get(req.lang)
    if not tts_service:
        raise HTTPException(status_code=404, detail=f"Language '{req.lang}' is not supported.")

    wav_bytes = tts_service.synthesize(req.text)

    # ─── Save a copy locally ─────────────────────────────────────────────────
    # 1. ensure your output folder exists
    save_dir = "tts_audios"
    os.makedirs(save_dir, exist_ok=True)

    # 2. pick a filename (e.g. include timestamp  lang)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    filename = f"tts_{req.lang}_{timestamp}.wav"
    filepath = os.path.join(save_dir, filename)

    # 3. write the bytes to disk
    with open(filepath, "wb") as f:
        f.write(wav_bytes)
    print(f"[saved] {filepath}")
    # ─────────────────────────────────────────────────────────────────────────

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f"inline; filename=tts_{req.lang}.wav"}
    )