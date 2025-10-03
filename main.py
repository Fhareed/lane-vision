import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from sort import Sort   # tracker

# ---------- Config ----------
VIDEO_PATH = "driving.mp4"
CONF_THRES = 0.6
CLASSES_OF_INTEREST = {2, 3, 5, 7}  # car, motorcycle, bus, truck
SMOOTH_FRAMES = 12
ROI_TOP_RATIO = 0.58
# ----------------------------

left_hist, right_hist = deque(maxlen=SMOOTH_FRAMES), deque(maxlen=SMOOTH_FRAMES)

def make_coordinates(frame, line_params):
    slope, intercept = line_params
    y1 = frame.shape[0]
    y2 = int(y1 * ROI_TOP_RATIO)
    if abs(slope) < 1e-3:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(frame, lines):
    left_fit, right_fit = [], []
    if lines is None:
        return []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        if abs(slope) < 0.3:
            continue
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    out = []
    if left_fit:
        l = np.mean(left_fit, axis=0)
        left_hist.append(l)
        l = np.mean(left_hist, axis=0)
        coords = make_coordinates(frame, l)
        if coords is not None: out.append(coords)
    if right_fit:
        r = np.mean(right_fit, axis=0)
        right_hist.append(r)
        r = np.mean(right_hist, axis=0)
        coords = make_coordinates(frame, r)
        if coords is not None: out.append(coords)
    return out

def draw_lane_poly(frame, lanes):
    if len(lanes) == 2:
        (x1l,y1l,x2l,y2l), (x1r,y1r,x2r,y2r) = lanes
        pts = np.array([[x1l,y1l],[x2l,y2l],[x2r,y2r],[x1r,y1r]], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0,255,0))
        return cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    return frame

# Load YOLO + SORT
yolo = YOLO("yolov8s.pt")   # or yolov8n.pt for speed
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]

    # ---- Lane detection ----
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white, upper_white = np.array([0,0,200]), np.array([255,40,255])
    lower_yel, upper_yel = np.array([15,100,100]), np.array([35,255,255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yel   = cv2.inRange(hsv, lower_yel, upper_yel)
    mask_lane  = cv2.bitwise_or(mask_white, mask_yel)
    lane_img   = cv2.bitwise_and(frame, frame, mask=mask_lane)
    gray = cv2.cvtColor(lane_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 75, 175)

    mask = np.zeros_like(edges)
    poly = np.array([[
        (int(0.08*w), h),
        (int(0.92*w), h),
        (int(0.58*w), int(ROI_TOP_RATIO*h)),
        (int(0.42*w), int(ROI_TOP_RATIO*h))
    ]], np.int32)
    cv2.fillPoly(mask, poly, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(roi, rho=2, theta=np.pi/180, threshold=100,
                            minLineLength=60, maxLineGap=200)
    lane_lines = average_slope_intercept(frame, lines)
    vis = draw_lane_poly(frame.copy(), lane_lines)
    for ln in lane_lines:
        x1,y1,x2,y2 = ln
        cv2.line(vis, (x1,y1), (x2,y2), (0,255,0), 8)

    # ---- YOLO + SORT ----
    results = yolo.predict(vis, conf=CONF_THRES, imgsz=640, device="mps", verbose=False)
    r = results[0]

    dets = []
    if r.boxes is not None:
        for b in r.boxes:
            cls = int(b.cls)
            if cls not in CLASSES_OF_INTEREST:
                continue
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            dets.append([x1,y1,x2,y2,conf])
    dets = np.array(dets)

    if dets.shape[0] > 0:
        tracks = tracker.update(dets)
    else:
        tracks = []

    for t in tracks:
        x1,y1,x2,y2,tid = map(int, t)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(vis, f"ID {tid}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Lane + YOLO + SORT", vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()