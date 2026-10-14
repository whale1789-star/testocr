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
    
    if isinstance(result, (int, float)):
        return {"status": "success", "score": result}
    else:
        return {"status": "error", "message": result}

# --- 輕量化視角校正演算法 ---
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

# --- 終極版：支援拍照透視校正 + 截圖直出 ---
def process_omr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # 1. 尋找畫面中所有的輪廓
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. 找出四個角落的「定位黑塊」
    markers = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        area = cv2.contourArea(c)
        # 黑塊特徵：大於20px、長寬比接近正方形、填滿率高
        if w >= 20 and h >= 20 and 0.6 <= ar <= 1.4:
            extent = area / float(w * h)
            if extent >= 0.7:  # 正方形的填滿率很高
                markers.append(c)
                
    # 依照面積由大到小排序，取出最大的 4 個方塊
    markers = sorted(markers, key=cv2.contourArea, reverse=True)[:4]
    
    warped_thresh = thresh
    
    # 3. 如果成功找到 4 個定位塊，就執行「視角校正 (拉平考卷)」
    if len(markers) == 4:
        pts = []
        for m in markers:
            M = cv2.moments(m)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                pts.append([cX, cY])
        pts = np.array(pts, dtype="float32")
        
        # 根據四個黑塊的中心點，把圖片拉成平整的矩形！
        warped_gray = get_four_point_transform(gray, pts)
        warped_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # 4. 在 (拉平後的) 影像中尋找 40 個選項泡泡
    cnts, _ = cv2.findContours(warped_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        area = cv2.contourArea(c)
        
        # 泡泡特徵篩選 (放寬標準以適應拍照)
        if 10 <= w <= 100 and 10 <= h <= 100 and 0.6 <= ar <= 1.4:
            extent = area / float(w * h)
            # 圓形的 extent 約為 0.78，空心圓圈會再低一點
            if 0.3 <= extent <= 0.9:
                questionCnts.append(c)
                
    if len(questionCnts) != 40:
        return f"拍照辨識失敗：找到了 {len(questionCnts)} 個泡泡 (應為 40 個)。請確保「四個角落的黑塊」都清晰入鏡且沒有被遮擋。"

    # 5. 排序題目 (因為圖片已經拉平了，現在排序絕對不會錯亂)
    questionCnts, _ = sort_contours(questionCnts, method="top-to-bottom")
    
    # 標準答案 (第 1~10 題皆為 A)
    ANSWER_KEY = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    correct = 0

    # 6. 批改計分
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        cnts = questionCnts[i:i + 4]
        cnts, _ = sort_contours(cnts, method="left-to-right")
        
        bubbled = None
        for (j, c) in enumerate(cnts):
            mask = np.zeros(warped_thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(warped_thresh, warped_thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        if ANSWER_KEY[q] == bubbled[1]:
            correct += 1

    return int((correct / 10.0) * 100)
