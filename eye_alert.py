# eye_alert_opencv.py
import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import os
import sys
import time
from collections import deque

# ---------- إعداد المسارات ----------
if getattr(sys, 'frozen', False):
    # إذا البرنامج ملف exe
    dir_path = sys._MEIPASS
else:
    # إذا تشغيله بالبايثون
    dir_path = os.path.dirname(os.path.realpath(__file__))

alarm_path = os.path.join(dir_path, "alarm.wav")  # ضع alarm.wav في نفس المجلد أو عدّل المسار

# ---------- Mediapipe ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# ---------- دالة EAR ----------
def eye_aspect_ratio(eye_points, landmarks):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_points]
    vert1 = np.linalg.norm(p[1] - p[5])
    vert2 = np.linalg.norm(p[2] - p[4])
    horiz = np.linalg.norm(p[0] - p[3])
    if horiz == 0:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

# ---------- الإعدادات ----------
ALERT_THRESHOLD = 0.22    # اعتبار العين مغلقة تمامًا
PARTIAL_THRESHOLD = 0.30
FULL_OPEN_THRESHOLD = 0.35
ALERT_DURATION = 0.9      # ثواني قبل تشغيل الإنذار
MIN_BLINK_DURATION = 0.06 # أقل مدة تُحسب كغمضة

# ---------- متغيرات الحالة والإحصاءات ----------
paused = False
blink_count = 0
longest_closure = 0.0
current_closure_start = None
in_closed_state = False

# لمعدل فتح العين
ear_values = []        # لتخزين جميع القيم لحساب المتوسط
ear_running_sum = 0.0
ear_samples = 0

# للرسم (graph)
GRAPH_WIDTH = 240
GRAPH_HEIGHT = 480
GRAPH_LEN = 200
graph_buffer = deque(maxlen=GRAPH_LEN)  # سيخزن نسب الـEAR (0-100)

# مؤقت/play object للصوت
play_obj = None

# دالة لتحويل EAR إلى نسبة 0-100
def ear_to_percentage(ear):
    ear_clipped = np.clip(ear, ALERT_THRESHOLD, FULL_OPEN_THRESHOLD)
    pct = (ear_clipped - ALERT_THRESHOLD) / (FULL_OPEN_THRESHOLD - ALERT_THRESHOLD) * 100.0
    return int(np.round(pct))

# ---------- فتح الكاميرا ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("EyeAlert", cv2.WINDOW_NORMAL)

try:
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            avg_ear = 0.0
            pct = 0

            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                left_ear = eye_aspect_ratio(LEFT_EYE, lm)
                right_ear = eye_aspect_ratio(RIGHT_EYE, lm)
                avg_ear = (left_ear + right_ear) / 2.0
                pct = ear_to_percentage(avg_ear)

                # تحديث إحصاءات المتوسط
                ear_values.append(pct)
                ear_running_sum += pct if 'ear_running_sum' in globals() else pct
                ear_samples += 1 if 'ear_samples' in globals() else 1

                # إضافة للنقطة البيانية
                graph_buffer.append(pct)

                # ---------- اكتشاف الإغلاق/الفتح ----------
                now = time.time()
                if avg_ear <= ALERT_THRESHOLD:
                    if not in_closed_state:
                        in_closed_state = True
                        current_closure_start = now
                    if current_closure_start is not None:
                        duration_closed = now - current_closure_start
                        if duration_closed >= ALERT_DURATION:
                            if play_obj is None or not getattr(play_obj, "is_playing", lambda: False)():
                                try:
                                    wave_obj = sa.WaveObject.from_wave_file(alarm_path)
                                    play_obj = wave_obj.play()
                                except Exception as e:
                                    play_obj = None
                else:
                    if in_closed_state:
                        now2 = time.time()
                        duration = now2 - (current_closure_start or now2)
                        if duration >= MIN_BLINK_DURATION:
                            blink_count += 1
                            if duration > longest_closure:
                                longest_closure = duration
                        if play_obj is not None and getattr(play_obj, "is_playing", lambda: False)():
                            try:
                                play_obj.stop()
                            except:
                                pass
                        play_obj = None
                    in_closed_state = False
                    current_closure_start = None

                # ---------------- رسم النقاط والمعلومات على الفيديو ----------------
                for idx in LEFT_EYE + RIGHT_EYE:
                    x = int(lm[idx].x * w)
                    y = int(lm[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0,255,0), -1)

                if ear_samples > 0:
                    average_open = sum(ear_values) / len(ear_values)
                else:
                    average_open = 0.0

                cv2.putText(frame, f"Eye: {pct}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"Blink count: {blink_count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
                cv2.putText(frame, f"Longest closure: {longest_closure:.2f}s", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,200), 2)
                cv2.putText(frame, f"Avg open: {average_open:.1f}%", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 2)

            else:
                graph_buffer.append(0)

        # ---------- رسم الـ Graph ----------
        vis_h = frame.shape[0]
        vis_w = GRAPH_WIDTH
        graph_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8) + 15

        if len(graph_buffer) > 1:
            pts = []
            for i, val in enumerate(graph_buffer):
                x = int((i / (GRAPH_LEN-1)) * (vis_w-10))
                y = int(vis_h - (val / 100.0) * (vis_h-20) - 10)
                pts.append((x+5, y))
            for i in range(1, len(pts)):
                cv2.line(graph_img, pts[i-1], pts[i], (0,200,0), 2)

            cv2.putText(graph_img, "Eye %", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.line(graph_img, (0, vis_h-10), (vis_w, vis_h-10), (60,60,60), 1)
            cv2.putText(graph_img, "0", (vis_w-20, vis_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            mid_y = int(vis_h - (50/100.0) * (vis_h-20) - 10)
            cv2.line(graph_img, (0, mid_y), (vis_w, mid_y), (50,50,50), 1)
            cv2.putText(graph_img, "50", (vis_w-30, mid_y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            top_y = int(vis_h - (100/100.0) * (vis_h-20) - 10)
            cv2.line(graph_img, (0, top_y), (vis_w, top_y), (50,50,50), 1)
            cv2.putText(graph_img, "100", (vis_w-40, top_y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

        combined = np.zeros((max(frame.shape[0], graph_img.shape[0]), frame.shape[1] + graph_img.shape[1], 3), dtype=np.uint8)
        combined[0:frame.shape[0], 0:frame.shape[1]] = frame
        combined[0:graph_img.shape[0], frame.shape[1]:frame.shape[1]+graph_img.shape[1]] = graph_img

        cv2.putText(combined, "p: pause/resume  |  r: reset stats  |  q/ESC: quit", (10, combined.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        cv2.imshow("EyeAlert", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            paused = not paused
            if play_obj is not None:
                try:
                    play_obj.stop()
                except:
                    pass
                play_obj = None
        elif key == ord('r'):
            blink_count = 0
            longest_closure = 0.0
            current_closure_start = None
            in_closed_state = False
            ear_values.clear()
            ear_running_sum = 0.0
            ear_samples = 0
            graph_buffer.clear()

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
