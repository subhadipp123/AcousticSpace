from fastapi import FastAPI

from app.api.routes import upload

app = FastAPI(title="AcousticSpace API")

app.include_router(upload.router)


@app.get("/")
def root():
    return {"status": "AcousticSpace API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}