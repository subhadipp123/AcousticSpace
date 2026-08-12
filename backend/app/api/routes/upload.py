import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ml.preprocessing import preprocess_audio

router = APIRouter()

UPLOAD_DIR = "data/uploads"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = preprocess_audio(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")

    return {
        "filename": file.filename,
        "duration_seconds": result["duration_seconds"],
        "rir_features": result["rir_features"],
    }