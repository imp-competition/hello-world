from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time
import math
import serial
import json, os

# =========================================================
# 기본 설정
# =========================================================
HFOV_DEG = 66.0
VFOV_DEG = 41.0
FRAME_W, FRAME_H = 1280, 720
TARGET_FPS = 30
MIRROR = True

AREA_FRAC_MIN = 0.0005
AREA_FRAC_MAX = 0.20
CIRC_MIN = 0.90

# 모폴로지
OPEN_K = 5
CLOSE_K = 7
OPEN_ITERS = 1
CLOSE_ITERS = 2

# 추적/모터
SMOOTH_ALPHA = 0.4
MIN_MOVE_DEG = 1.0          # 이 값보다 작은 스텝은 무시(양자화)
DEADZONE_PX  = 8
RANGE_CM     = 200.0
H_CM_PER_DEG = 1.32
V_CM_PER_DEG = 1.32

# ======= PD 제어 관련 (튜닝 키 포함) =======
KpX, KpY = 1.00, 1.00       # 비례 이득(수평/수직)
KvX, KvY = 0.00, 0.00       # 속도(미분) 이득(수평/수직)
STEP_K   = 0.05             # K 값 증감 단위
MAX_STEP_DEG = 10           # 프레임당 각속도 제한(모터 명령 최대 변화량)

# UART
UART_PAN  = '/dev/ttyAMA2'
UART_TILT = '/dev/ttyAMA3'
BAUDRATE  = 115200
SER_TIMEOUT = 0.1

# 서보
SERVO_MIN = 0
SERVO_MAX = 180
SERVO_CENTER_PAN  = 90
SERVO_CENTER_TILT = 90
SIGN_PAN  = +1
SIGN_TILT = +1   # 필요시 -1로 바꿔 테스트 가능

# 한계(틸트 45~180)
PAN_MIN,  PAN_MAX  = 0, 180
TILT_MIN, TILT_MAX = 45, 180

# =========================================================
# 점수 가중치 (원형 + 색)
# =========================================================
W_CIRC  = 0.40
W_COLOR = 0.60

# 색 내부 가중 (S 최우선)
WC_H = 0.15
WC_S = 0.65
WC_V = 0.20

# =========================================================
# 목표 색 구간(네가 관측한 범위)
# =========================================================
H_LO, H_HI = 3.0, 11.0
S_LO, S_HI = 180.0, 199.0   # 필요하면 179.0로 낮춰도 됨
V_LO, V_HI = 175.0, 199.0

H_SIGMA_OUT = 6.0    # deg
S_SIGMA_OUT = 15.0   # S 단위
V_SIGMA_OUT = 15.0   # V 단위

# =========================================================
# 총구 좌표 & 영점(Zero Offset)
# =========================================================
MUZZLE_PX = 640
MUZZLE_PY = 600

PAN_OFFSET_INIT  = 119
TILT_OFFSET_INIT = 98
PAN_OFFSET  = PAN_OFFSET_INIT
TILT_OFFSET = TILT_OFFSET_INIT

ADJ_STEP = 1
SETTINGS = "/home/pi/zero_offsets.json"

# =========================================================
# 마스크 생성 (빨강 계열 후보)
# =========================================================
def make_mask_from_bgr(frame_bgr, use_clahe=True):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)

    if use_clahe:
        h, s, v = cv.split(hsv)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        hsv = cv.merge([h, s, v])

    def or_ranges(hsv_img, ranges):
        m = None
        for lo, hi in ranges:
            cur = cv.inRange(hsv_img, np.array(lo), np.array(hi))
            m = cur if m is None else cv.bitwise_or(m, cur)
        return m

    # 빨강(밝음/어두움) — 조명 견고성
    red_bright = [((0,130,80),(15,255,255)), ((165,130,80),(180,255,255))]
    red_dark   = [((0, 90,40),(15,255,255)), ((165, 90,40),(180,255,255))]
    m_red = cv.bitwise_or(or_ranges(hsv, red_bright), or_ranges(hsv, red_dark))

    mask = m_red

    k_open  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  k_open,  iterations=OPEN_ITERS)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k_close, iterations=CLOSE_ITERS)
    return mask

# =========================================================
# 기하/보조
# =========================================================
def circularity_from_contour(cnt):
    area = cv.contourArea(cnt)
    perim = cv.arcLength(cnt, True)
    if perim <= 0:
        return 0.0, area
    circ = 4.0 * math.pi * area / (perim * perim)
    return circ, area

def px_to_deg(dx_px, dy_px, width, height):
    deg_x = (dx_px / float(width))  * HFOV_DEG
    deg_y = (dy_px / float(height)) * VFOV_DEG
    return deg_x, deg_y

def angle_to_cm(deg_x, deg_y, range_cm):
    dx_cm = range_cm * math.tan(math.radians(deg_x))
    dy_cm = range_cm * math.tan(math.radians(deg_y))
    return dx_cm, dy_cm

def mean_hsv_in_hull(frame_bgr, hull):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    poly = hull.reshape(-1, 2)
    cv.fillConvexPoly(mask, poly, 255)
    m = mask > 0
    if not np.any(m):
        return None
    h, s, v = cv.split(hsv)
    Hm = float(np.mean(h[m]))
    Sm = float(np.mean(s[m]))
    Vm = float(np.mean(v[m]))
    return (Hm, Sm, Vm)

# 구간 점수: 안이면 1.0, 밖이면 가우시안으로 감쇠
def band_score(val, lo, hi, sigma_out):
    if lo <= val <= hi:
        return 1.0
    d = (lo - val) if val < lo else (val - hi)
    return math.exp(-0.5 * (d / max(1e-6, sigma_out))**2)

# 색 점수(구간 기반, S에 높은 가중)
def color_score_band(Hm, Sm, Vm):
    scH = band_score(Hm, H_LO, H_HI, H_SIGMA_OUT)
    scS = band_score(Sm, S_LO, S_HI, S_SIGMA_OUT)
    scV = band_score(Vm, V_LO, V_HI, V_SIGMA_OUT)
    num = WC_H*scH + WC_S*scS + WC_V*scV
    den = WC_H + WC_S + WC_V
    return max(0.0, min(1.0, num/den)), (scH, scS, scV)

# 후보 선택(원형 + 색)
def pick_best_target(mask, frame_area_min, frame_area_max, circ_min, frame_bgr=None):
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best = None

    for c in cnts:
        hull = cv.convexHull(c)
        circ, area = circularity_from_contour(hull)
        if area < frame_area_min or area > frame_area_max:
            continue
        if circ < circ_min:
            continue

        x, y, w, h = cv.boundingRect(hull)
        M = cv.moments(hull)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        hsv_mean = None
        col_sc = 0.0
        scH = scS = scV = 0.0
        if frame_bgr is not None:
            hsv_mean = mean_hsv_in_hull(frame_bgr, hull)
            if hsv_mean is not None:
                Hm, Sm, Vm = hsv_mean
                col_sc, (scH, scS, scV) = color_score_band(Hm, Sm, Vm)

        total_sc = (W_CIRC * circ) + (W_COLOR * col_sc)

        cand = {
            "score": total_sc,
            "score_breakdown": {
                "circ": circ, "color": col_sc, "H": scH, "S": scS, "V": scV
            },
            "circularity": circ,
            "area": int(area),
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "hull": hull,
            "hsv_mean": hsv_mean,
        }

        if (best is None) or (cand["score"] > best["score"]) or \
           (abs(cand["score"] - best["score"]) < 1e-6 and cand["area"] > best["area"]):
            best = cand
    return best

# 상대각(±) → 절대 서보각(0~180)
def delta_to_servo_angles(cmd_deg_x, cmd_deg_y):
    pan  = SERVO_CENTER_PAN  + SIGN_PAN  * cmd_deg_x
    tilt = SERVO_CENTER_TILT + SIGN_TILT * cmd_deg_y
    pan  = max(PAN_MIN,  min(PAN_MAX,  int(pan)))
    tilt = max(TILT_MIN, min(TILT_MAX, int(tilt)))
    return pan, tilt

# =========================================================
# 오프셋/게인 저장/로드
# =========================================================
def save_offsets():
    try:
        with open(SETTINGS, "w") as f:
            json.dump({
                "PAN_OFFSET":  PAN_OFFSET,
                "TILT_OFFSET": TILT_OFFSET,
                "ADJ_STEP":    ADJ_STEP,
                "KpX": KpX, "KpY": KpY,
                "KvX": KvX, "KvY": KvY,
                "STEP_K": STEP_K,
                "MAX_STEP_DEG": MAX_STEP_DEG
            }, f, indent=2)
        print("[SAVE] OK:", SETTINGS)
    except Exception as e:
        print("[SAVE] FAIL:", e)

def load_offsets():
    global PAN_OFFSET, TILT_OFFSET, ADJ_STEP
    global KpX, KpY, KvX, KvY, STEP_K, MAX_STEP_DEG
    try:
        if os.path.exists(SETTINGS):
            d = json.load(open(SETTINGS, "r"))
            PAN_OFFSET  = int(d.get("PAN_OFFSET",  PAN_OFFSET))
            TILT_OFFSET = int(d.get("TILT_OFFSET", TILT_OFFSET))
            ADJ_STEP    = int(d.get("ADJ_STEP",    ADJ_STEP))
            KpX = float(d.get("KpX", KpX)); KpY = float(d.get("KpY", KpY))
            KvX = float(d.get("KvX", KvX)); KvY = float(d.get("KvY", KvY))
            STEP_K = float(d.get("STEP_K", STEP_K))
            MAX_STEP_DEG = int(d.get("MAX_STEP_DEG", MAX_STEP_DEG))
            print("[LOAD] OK", SETTINGS, "=>",
                  f"pan:{PAN_OFFSET} tilt:{TILT_OFFSET} step:{ADJ_STEP} | "
                  f"Kp({KpX:.2f},{KpY:.2f}) Kv({KvX:.2f},{KvY:.2f}) MAX:{MAX_STEP_DEG}")
    except Exception as e:
        print("[LOAD] FAIL:", e)

# =========================================================
# UART
# =========================================================
class MotorUART:
    def __init__(self, port_pan, port_tilt, baud, timeout):
        self.ser_pan  = serial.Serial(port_pan,  baud, timeout=timeout)
        self.ser_tilt = serial.Serial(port_tilt, baud, timeout=timeout)
        time.sleep(0.2)
        self.last_pan_angle  = None
        self.last_tilt_angle = None

    def send_servo(self, pan_angle, tilt_angle):
        pan_angle  = max(PAN_MIN,  min(PAN_MAX,  int(pan_angle)))
        tilt_angle = max(TILT_MIN, min(TILT_MAX, int(tilt_angle)))
        if pan_angle != self.last_pan_angle:
            self.ser_pan.write(f"{pan_angle}\n".encode('ascii'))
            self.last_pan_angle = pan_angle
        if tilt_angle != self.last_tilt_angle:
            self.ser_tilt.write(f"{tilt_angle}\n".encode('ascii'))
            self.last_tilt_angle = tilt_angle

    def close(self):
        for s in (self.ser_pan, self.ser_tilt):
            try: s.close()
            except: pass

# =========================================================
# UI 유틸
# =========================================================
def draw_text(vis, text, org, color=(0,255,255), scale=0.7):
    cv.putText(vis, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4, cv.LINE_AA)
    cv.putText(vis, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv.LINE_AA)

# =========================================================
# 메인
# =========================================================
def main():
    global PAN_OFFSET, TILT_OFFSET, ADJ_STEP
    global KpX, KpY, KvX, KvY, MAX_STEP_DEG

    load_offsets()  # 시작 시 저장된 설정 로드
    motors = MotorUART(UART_PAN, UART_TILT, BAUDRATE, SER_TIMEOUT)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H)})
    picam2.configure(config)
    picam2.start()

    cx0, cy0 = FRAME_W // 2, FRAME_H // 2
    ema_cx, ema_cy = None, None

    frame_area = FRAME_W * FRAME_H
    AREA_MIN = int(AREA_FRAC_MIN * frame_area)
    AREA_MAX = int(AREA_FRAC_MAX * frame_area)

    prev_t = time.time()
    fps = 0.0

    # PD용 이전 오차(모터각 단위) 저장
    prev_err_deg_x = 0.0
    prev_err_deg_y = 0.0

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            if MIRROR:
                frame = cv.flip(frame, 1)

            mask = make_mask_from_bgr(frame, use_clahe=True)
            target = pick_best_target(mask, AREA_MIN, AREA_MAX, CIRC_MIN, frame_bgr=frame)

            vis = frame.copy()
            # 중앙 십자 + 총구 기준점 마커
            cv.drawMarker(vis, (cx0, cy0), (255, 255, 255), markerType=cv.MARKER_CROSS, markerSize=24, thickness=2)
            cv.circle(vis, (cx0, cy0), 3, (0, 0, 255), -1)
            cv.drawMarker(vis, (MUZZLE_PX, MUZZLE_PY), (0,255,255), markerType=cv.MARKER_TILTED_CROSS, markerSize=24, thickness=2)
            if MIRROR:
                draw_text(vis, "MIRROR: ON", (FRAME_W - 180, 30), color=(255,255,255), scale=0.7)

            now = time.time()
            dt  = max(1e-3, now - prev_t)  # 보호
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            if target is not None:
                (bx, by, bw, bh) = target["bbox"]
                (cx, cy) = target["center"]
                area = target["area"]
                circ = target["circularity"]
                hull = target["hull"]

                if ema_cx is None:
                    ema_cx, ema_cy = cx, cy
                else:
                    ema_cx = int(SMOOTH_ALPHA * cx + (1 - SMOOTH_ALPHA) * ema_cx)
                    ema_cy = int(SMOOTH_ALPHA * cy + (1 - SMOOTH_ALPHA) * ema_cy)

                # === 기준점 변경: 중앙 → 총구좌표
                dx_px = ema_cx - MUZZLE_PX
                dy_px = ema_cy - MUZZLE_PY
                if abs(dx_px) < DEADZONE_PX: dx_px = 0
                if abs(dy_px) < DEADZONE_PX: dy_px = 0

                # px → deg → cm → "요구 모터각(연속값)"
                deg_x, deg_y = px_to_deg(dx_px, dy_px, FRAME_W, FRAME_H)
                dx_cm, dy_cm = angle_to_cm(deg_x, deg_y, RANGE_CM)
                err_deg_x = dx_cm / H_CM_PER_DEG   # 모터가 움직여야 할 각(연속값)
                err_deg_y = dy_cm / V_CM_PER_DEG

                # ===== PD 제어 =====
                derr_x = (err_deg_x - prev_err_deg_x) / dt
                derr_y = (err_deg_y - prev_err_deg_y) / dt
                pd_x = KpX * err_deg_x + KvX * derr_x
                pd_y = KpY * err_deg_y + KvY * derr_y

                # 한 프레임당 최대 변화량 제한
                def clamp_step(v, vmax):
                    if v > 0:  return  min(v, vmax)
                    else:      return -min(abs(v), vmax)

                pd_x = clamp_step(pd_x, MAX_STEP_DEG)
                pd_y = clamp_step(pd_y, MAX_STEP_DEG)

                # 양자화(작으면 0)
                def quantize(v):
                    s = 1 if v >= 0 else -1
                    mag = abs(v)
                    if mag < MIN_MOVE_DEG: return 0
                    return int(round(s * mag))

                cmd_deg_x = quantize(pd_x)
                cmd_deg_y = quantize(pd_y)

                prev_err_deg_x = err_deg_x
                prev_err_deg_y = err_deg_y

                servo_pan, servo_tilt = delta_to_servo_angles(cmd_deg_x, cmd_deg_y)

                # === 영점 보정 적용
                servo_pan  = PAN_OFFSET  + (servo_pan  - SERVO_CENTER_PAN)
                servo_tilt = TILT_OFFSET + (servo_tilt - SERVO_CENTER_TILT)
                servo_pan  = max(PAN_MIN,  min(PAN_MAX,  int(servo_pan)))
                servo_tilt = max(TILT_MIN, min(TILT_MAX, int(servo_tilt)))

                motors.send_servo(servo_pan, servo_tilt)

                # 시각화
                cv.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv.circle(vis, (ema_cx, ema_cy), 5, (0, 0, 255), -1)
                cv.drawContours(vis, [hull], -1, (255, 0, 255), 2)

                y0 = 28
                dy = 26
                draw_text(vis, f"area:{area}  circ:{circ:.3f}",
                          (bx, max(0, by - 10)), color=(0,255,0), scale=0.6)
                draw_text(vis, f"offset_px(mu):   ({dx_px:+d}, {dy_px:+d})", (10, y0), color=(50,220,50))
                draw_text(vis, f"offset_deg:      ({deg_x:+.2f}, {deg_y:+.2f})", (10, y0+dy), color=(50,220,50))
                draw_text(vis, f"err(motor°):     x={err_deg_x:+.2f}, y={err_deg_y:+.2f}", (10, y0+2*dy), color=(200,230,255))
                draw_text(vis, f"PD out°:         x={pd_x:+.2f}, y={pd_y:+.2f}", (10, y0+3*dy), color=(0,255,255))
                draw_text(vis, f"motor_cmdΔ:      pan={cmd_deg_x:+d}°, tilt={cmd_deg_y:+d}°",
                          (10, y0+4*dy), color=(0,255,255))
                draw_text(vis, f"servo_out(Zero): pan={servo_pan:3d}°, tilt={servo_tilt:3d}°",
                          (10, y0+5*dy), color=(0,255,255))

                if target["hsv_mean"] is not None:
                    Hm, Sm, Vm = target["hsv_mean"]
                    scb = target["score_breakdown"]

                    draw_text(vis, f"HSV(mean): H={Hm:.2f}, S={Sm:.2f}, V={Vm:.2f}",
                              (10, y0+7*dy), color=(200,255,200))
                    draw_text(vis, f"Band H[{H_LO:.0f},{H_HI:.0f}] S[{S_LO:.0f},{S_HI:.0f}] V[{V_LO:.0f},{V_HI:.0f}]",
                              (10, y0+8*dy), color=(255,255,255), scale=0.62)
                    draw_text(vis, f"color(H,S,V)=({scb['H']:.3f},{scb['S']:.3f},{scb['V']:.3f})  "
                                   f"=> color={scb['color']:.3f}",
                              (10, y0+9*dy), color=(200,255,200), scale=0.62)
                    draw_text(vis, f"total = {W_CIRC:.2f}*circ({scb['circ']:.3f}) + "
                                   f"{W_COLOR:.2f}*color({scb['color']:.3f}) = {target['score']:.3f}",
                              (10, y0+10*dy), color=(255,255,0), scale=0.62)

                    # 스와치(평균 HSV → BGR)
                    sw_hsv = np.uint8([[[Hm, Sm, Vm]]])
                    sw_bgr = cv.cvtColor(sw_hsv, cv.COLOR_HSV2BGR)[0,0].tolist()
                    x_sw, y_sw = 10, y0+11*dy-18
                    cv.rectangle(vis, (x_sw, y_sw), (x_sw+46, y_sw+20), (0,0,0), -1)
                    cv.rectangle(vis, (x_sw+3, y_sw+3), (x_sw+43, y_sw+17), sw_bgr, -1)
                    cv.rectangle(vis, (x_sw+3, y_sw+3), (x_sw+43, y_sw+17), (30,30,30), 1)

            else:
                ema_cx = ema_cy = None
                draw_text(vis, "No target", (10, 30), color=(0,0,255), scale=0.8)
                # 필요 시 중립복귀 로직을 사용하려면 아래 주석 해제
                # motors.send_servo(SERVO_CENTER_PAN + (PAN_OFFSET - SERVO_CENTER_PAN),
                #                  SERVO_CENTER_TILT + (TILT_OFFSET - SERVO_CENTER_TILT))

            # FPS & HUD
            draw_text(vis, f"FPS: {fps:.1f}", (10, FRAME_H - 10), color=(255,255,0), scale=0.8)

            # 하단 HUD: 오프셋/게인/최대스텝
            draw_text(vis, f"ZERO pan:{PAN_OFFSET:+d}° tilt:{TILT_OFFSET:+d}° (step={ADJ_STEP}°) | "
                           f"Kp({KpX:.2f},{KpY:.2f}) Kv({KvX:.2f},{KvY:.2f}) MAX_STEP:{MAX_STEP_DEG}°",
                      (FRAME_W-720, FRAME_H-12), color=(200,200,255), scale=0.6)

            cv.imshow("frame", vis)
            cv.imshow("mask", mask)

            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

            # === 영점 조정 키
            if   key==ord('u'): TILT_OFFSET-=ADJ_STEP; print("[OFFSET] TILT",TILT_OFFSET)
            elif key==ord('d'): TILT_OFFSET+=ADJ_STEP; print("[OFFSET] TILT",TILT_OFFSET)
            elif key==ord('r'): PAN_OFFSET +=ADJ_STEP; print("[OFFSET] PAN",PAN_OFFSET)
            elif key==ord('l'): PAN_OFFSET -=ADJ_STEP; print("[OFFSET] PAN",PAN_OFFSET)
            elif key==ord('c'):
                PAN_OFFSET,TILT_OFFSET=PAN_OFFSET_INIT,TILT_OFFSET_INIT
                print("[OFFSET] reset", PAN_OFFSET, TILT_OFFSET)

            # 숫자키 튜닝(축별) — Kp/Kv/MAX_STEP
            elif key==ord('1'): KpX=max(0.0,round(KpX-STEP_K,3)); print("[KpX]",KpX)
            elif key==ord('2'): KpX=round(KpX+STEP_K,3);         print("[KpX]",KpX)
            elif key==ord('3'): KpY=max(0.0,round(KpY-STEP_K,3)); print("[KpY]",KpY)
            elif key==ord('4'): KpY=round(KpY+STEP_K,3);         print("[KpY]",KpY)
            elif key==ord('5'): KvX=max(0.0,round(KvX-STEP_K,3)); print("[KvX]",KvX)
            elif key==ord('6'): KvX=round(KvX+STEP_K,3);         print("[KvX]",KvX)
            elif key==ord('7'): KvY=max(0.0,round(KvY-STEP_K,3)); print("[KvY]",KvY)
            elif key==ord('8'): KvY=round(KvY+STEP_K,3);         print("[KvY]",KvY)
            elif key==ord('9'): MAX_STEP_DEG=max(1,MAX_STEP_DEG-1); print("[MAX_STEP_DEG]",MAX_STEP_DEG)
            elif key==ord('0'): MAX_STEP_DEG=min(30,MAX_STEP_DEG+1); print("[MAX_STEP_DEG]",MAX_STEP_DEG)

            # 저장/로드 & 스텝
            elif key==ord('['): ADJ_STEP=max(1,ADJ_STEP-1); print("[STEP]",ADJ_STEP)
            elif key==ord(']'): ADJ_STEP+=1; print("[STEP]",ADJ_STEP)
            elif key==ord('s'): save_offsets()
            elif key==ord('o'): load_offsets()

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cv.destroyAllWindows()
        picam2.stop()
        motors.close()

if __name__ == "__main__":
    main()
