from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils import predict
import shutil
import os
import uuid

# initilizing Fast API
app = FastAPI()

# connect to all other tech Ex. React,Angular,HTML, JS, Spring Boot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allowed origins
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, PUT, DELETE
    allow_headers=["*"],          # All headers
)

@app.get("/")
def home():
    return {"message": "Pneumonia Detection API is Running 🚀"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    
    # Unique temp file
    file_location = f"temp_{uuid.uuid4().hex}.jpg"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict(file_location)
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

    return {
        "filename": file.filename,
        "result": result
    }