from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

# 就是這一行！Uvicorn 正在找這個名叫 'app' 的變數
app = FastAPI() 

@app.post("/api/scan")
async def scan_card(file: UploadFile = File(...)):
    # ...下方的程式碼...
