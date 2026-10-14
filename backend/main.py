from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def wake_up():
    return {"message": "Server is awake!"}

@app.post("/api/scan")
async def scan_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    result = process_omr(img)
    
    # 如果回傳的是整數代表成功，字串代表錯誤訊息
    if isinstance(result, int) or isinstance(result, float):
        return {"status": "success", "score": result}
    else:
        return {"status": "error", "message": result}

# --- 輕量化視角校正 ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# --- 輪廓排序輔助函式 ---
def sort_contours(cnts, method="top-to-bottom"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    return (list(cnts), list(boundingBoxes))

# --- 真實 OpenCV 閱卷邏輯 ---
def process_omr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 1. 尋找答案卡的最外框
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is None:
        return "找不到四個角落的黑塊，請確保答案卡邊框清晰且都在畫面內。"

    # 2. 視角校正
    pts = docCnt.reshape(4, 2)
    warped = get_four_point_transform(gray, pts)

    # 3. 圖片二值化 (塗黑的地方變白，背景變黑)
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 4. 尋找所有的圓形泡泡
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # 過濾出形狀接近正方形/圓形的輪廓
        if w >= 15 and h >= 15 and ar >= 0.7 and ar <= 1.3:
            questionCnts.append(c)

    if len(questionCnts) != 40:
        return f"無法準確辨識選項泡泡。系統找到了 {len(questionCnts)} 個，但應為 40 個(10題x4選項)。請檢查反光或拍攝角度。"

    # 5. 由上到下排序題目
    questionCnts, _ = sort_contours(questionCnts, method="top-to-bottom")

    # 6. 設定標準答案 (這裡預設第 1~10 題全部都是 A 選項(0))
    # 選項對應：0=A, 1=B, 2=C, 3=D
    ANSWER_KEY = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    correct = 0

    # 7. 逐題批改 (每次取 4 個泡泡為一題)
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        # 取出這一題的 4 個選項，由左到右排序
        cnts = questionCnts[i:i + 4]
        cnts, _ = sort_contours(cnts, method="left-to-right")
        
        bubbled = None

        # 檢查這 4 個選項，哪一個被塗得最黑
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        # 比對標準答案
        if ANSWER_KEY[q] == bubbled[1]:
            correct += 1

    # 8. 計算最終分數
    score = int((correct / 10.0) * 100)
    return score
