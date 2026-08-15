from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import upload
from app.services.history import load_history


app = FastAPI(
    title="AcousticSpace API",
)


# Allow both localhost and 127.0.0.1 frontend URLs.
# The browser may treat these as different origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve generated spectrograms and other generated files.
app.mount(
    "/generated",
    StaticFiles(
        directory="data/generated"
    ),
    name="generated",
)


# Serve uploaded audio files.
app.mount(
    "/uploads",
    StaticFiles(
        directory="data/uploads"
    ),
    name="uploads",
)


# Upload and analysis API routes.
app.include_router(
    upload.router
)


@app.get("/")
def root():
    return {
        "status": "AcousticSpace API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/history")
def history():
    return {
        "items": load_history()
    }