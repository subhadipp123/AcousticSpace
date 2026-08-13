import os
import shutil

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)

from app.ml.preprocessing import (
    preprocess_audio,
    generate_spectrogram,
)

from app.ml.inference import predict_audio

from app.ml.segment_analysis import (
    analyze_segments,
)

from app.services.history import (
    save_analysis,
)


router = APIRouter()


UPLOAD_DIR = "data/uploads"
GENERATED_DIR = "data/generated"

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
}


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided",
        )

    safe_filename = os.path.basename(
        file.filename
    )

    ext = os.path.splitext(
        safe_filename
    )[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type "
                f"'{ext}'. "
                f"Allowed: "
                f"{ALLOWED_EXTENSIONS}"
            ),
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True,
    )

    os.makedirs(
        GENERATED_DIR,
        exist_ok=True,
    )

    save_path = os.path.join(
        UPLOAD_DIR,
        safe_filename,
    )

    try:
        with open(
            save_path,
            "wb",
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        acoustic_result = preprocess_audio(
            save_path
        )

        model_results = predict_audio(
            save_path
        )

        spectrogram_file = (
            generate_spectrogram(
                save_path,
                GENERATED_DIR,
            )
        )

        spectrogram_filename = (
            os.path.basename(
                spectrogram_file
            )
        )

        segments = analyze_segments(
            save_path
        )

        history_item = save_analysis(
            safe_filename,
            model_results["primary"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to process audio: "
                f"{str(e)}"
            ),
        )

    return {
        "filename": safe_filename,

        "duration_seconds":
            acoustic_result[
                "duration_seconds"
            ],

        "audio_url":
            f"/uploads/{safe_filename}",

        "rir_features":
            acoustic_result[
                "rir_features"
            ],

        "cnn":
            model_results["cnn"],

        "ast":
            model_results["ast"],

        "primary_prediction":
            model_results["primary"],

        "segments":
            segments,

        "spectrogram_path":
            (
                f"/generated/"
                f"{spectrogram_filename}"
            ),

        "history_item":
            history_item,
    }