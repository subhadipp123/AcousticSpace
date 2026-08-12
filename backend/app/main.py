from fastapi import FastAPI

app = FastAPI(title="AcousticSpace API")

@app.get("/")
def root():
    return {"status": "AcousticSpace API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}