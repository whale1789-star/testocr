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

# --- 全新升級：數位畫面直出閱卷演算法 ---
def process_omr(image):
    # 1. 轉灰階與高斯模糊去雜訊 (抵抗拍螢幕時的水波紋)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 自動二值化適應不同螢幕亮度
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # 3. 放棄尋找考卷外框，直接在全圖地毯式搜索所有輪廓
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        area = cv2.contourArea(c)
        
        # 篩選條件：長寬不能太小，也不能大到像整個螢幕，且形狀需接近正方形(外接框)
        if 10 <= w <= 150 and 10 <= h <= 150 and 0.6 <= ar <= 1.4:
            # 幾何過濾：圓形的面積約佔外接矩形的 78%，方形則是 100%。
            # 利用這點可以直接排除四個角落的黑塊，只留下圓圈！
            extent = area / float(w * h)
            if 0.4 <= extent <= 0.85:
                questionCnts.append(c)
                
    if len(questionCnts) != 40:
        return f"找到 {len(questionCnts)} 個泡泡，但需要 40 個。若您是用手機拍攝電腦螢幕，摩爾紋會破壞畫面。強烈建議：直接在手機上用無痕模式開啟網頁，點擊作答後「手機直接截圖」上傳！"

    # 4. 排序題目 (由上到下)
    questionCnts, _ = sort_contours(questionCnts, method="top-to-bottom")
    
    # 標準答案 (第 1~10 題皆為 A)
    ANSWER_KEY = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    correct = 0

    # 5. 批改
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        cnts = questionCnts[i:i + 4]
        cnts, _ = sort_contours(cnts, method="left-to-right")
        
        bubbled = None
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        if ANSWER_KEY[q] == bubbled[1]:
            correct += 1

    return int((correct / 10.0) * 100)
