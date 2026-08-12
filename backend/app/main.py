from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import upload

app = FastAPI(title="AcousticSpace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)


@app.get("/")
def root():
    return {"status": "AcousticSpace API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}