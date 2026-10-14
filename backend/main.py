# ... (上半部的 FastAPI 設定與 get_four_point_transform 保留) ...

# --- 新增：輪廓排序輔助函式 ---
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
    return (cnts, boundingBoxes)

# --- OpenCV 核心邏輯 (完整版) ---
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
        return "錯誤：無法辨識答案卡邊框"

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
        # 根據長寬比過濾出圓形的泡泡 (寬高大於 15px，比例接近 1:1)
        if w >= 15 and h >= 15 and ar >= 0.7 and ar <= 1.3:
            questionCnts.append(c)

    # 檢查是否剛好抓到 40 個選項 (10題 * 4選項)
    if len(questionCnts) != 40:
        return f"辨識失敗：找到 {len(questionCnts)} 個泡泡，格式不符 (應為 40 個)。請確保拍攝清晰無反光。"

    # 5. 由上到下排序泡泡 (將題目排序為 1~10 題)
    questionCnts, _ = sort_contours(questionCnts, method="top-to-bottom")

    # 6. 設定標準答案 (0=A, 1=B, 2=C, 3=D)
    # 這裡設定第 1~10 題的正確答案全部都是 A (索引值 0)
    ANSWER_KEY = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    correct = 0

    # 7. 逐題批改 (每次取 4 個泡泡為一題)
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        # 取出這一題的 4 個選項，並由左到右排序 (A, B, C, D)
        cnts = questionCnts[i:i + 4]
        cnts, _ = sort_contours(cnts, method="left-to-right")
        
        bubbled = None

        # 檢查這 4 個選項，哪一個塗得最黑 (白色像素最多)
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        # 比對這題最黑的選項與標準答案是否一致
        if ANSWER_KEY[q] == bubbled[1]:
            correct += 1

    # 8. 計算最終分數 (一題 10 分)
    score = int((correct / 10.0) * 100)
    return score
