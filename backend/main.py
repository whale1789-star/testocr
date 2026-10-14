from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform

app = FastAPI()

# 解決前端連線時的 CORS 錯誤 (跨來源資源共用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有網域連線 (包含你的 GitHub Pages)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 用於 cron-job.org 的自動喚醒路由
@app.get("/")
async def wake_up():
    return {"message": "Server is awake!"}

# 接收前端圖片並進行辨識的 API
@app.post("/api/scan")
async def scan_card(file: UploadFile = File(...)):
    # 1. 讀取前端傳來的圖片檔案轉為 OpenCV 格式
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. 呼叫 OMR 辨識邏輯
    score = process_omr(img)
    
    return {"status": "success", "score": score}

# OpenCV 核心邏輯
def process_omr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 1. 尋找答案卡的最外框 (邊緣檢測)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4: # 找到四個角落的矩形
                docCnt = approx
                break

    if docCnt is None:
        return "無法辨識邊框"

    # 2. 視角校正 (拉直變形與傾斜的卡片)
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # 3. 圖片二值化 (將塗黑區域變成純白，背景變黑，方便計算)
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 這裡會放入真實的選項座標比對邏輯
    # 目前先回傳一個測試用分數，確認前後端溝通順暢
    score = 85 

    return score
