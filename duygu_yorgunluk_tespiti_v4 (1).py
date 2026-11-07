import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

CALIB_FILE = "calibration.json"

# ======== JSON YÖNETİMİ ========
def load_all_users():
    
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Uyarı: {CALIB_FILE} dosyası bozuk. Yeni dosya oluşturuluyo.")
            return {"users": {}}
    return {"users": {}}

def save_all_users(data):
    
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ========FONKSİYONLAR ========
def find_landmarks(index, frame, face_landmarks):
   
    lm = face_landmarks.landmark[index]
    h, w, _ = frame.shape
    return np.array([lm.x * w, lm.y * h])

def euclidean_dist(p1, p2):
   
    return np.linalg.norm(p1 - p2)

def calculate_ear(face_landmarks, frame, is_right=False):
  
    if is_right:
        idx = [362, 385, 387, 263, 373, 380]
    else:
        idx = [33, 160, 158, 133, 153, 144]
    
    p = [find_landmarks(i, frame, face_landmarks) for i in idx]
    v1 = euclidean_dist(p[1], p[5])
    v2 = euclidean_dist(p[2], p[4])
    h1 = euclidean_dist(p[0], p[3])
    
    return (v1 + v2) / (2.0 * h1)

# ========  TÜRKÇE KARAKTERİ DÜZELTME ========
turkish_mapping = str.maketrans("İıŞşÇçĞğÜüÖö", "IiSsCcGgUuOo")
is_name_correct = False
user_name = ""

while not is_name_correct:
    raw_name = input("Kullanici Adini Girin:").strip()
    if len(raw_name) < 2:
        print("Lutfen gecerli bir isim giriniz.")
        continue
    
    user_name = raw_name.translate(turkish_mapping)
    is_name_correct = True


db = load_all_users()
user_exists = user_name in db["users"]
calib = db["users"].get(user_name, None)

# === VERİ KONTROLÜ  ===
required_keys = ["head_pitch_base", "brow_dist_base", "lip_ratio_base"] 
if user_exists and any(key not in calib for key in required_keys): 
    print(f"Uyarı: '{user_name}' için eski veri  tespit edildi Yeniden kalibrasyon yapılıyo")
    user_exists = False
    calib = None


# ======== MEDIAPIPE AYARLARI ========
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

# ======== KAMERA ========
cap = cv2.VideoCapture(0)
collecting_data = []
calib_start_time = None


# Uyku 
SLEEP_SMOOTHING_FACTOR = 0.03 
sleep_pct = 0.0 

# Stres 
STRESS_SMOOTHING_FACTOR = 0.05
stress_pct = 0.0
# =======================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # --- 1. TEMEL---
        face_left = find_landmarks(234, frame, face_landmarks)
        face_right = find_landmarks(454, frame, face_landmarks)
        face_width = euclidean_dist(face_left, face_right)

        # --- 2. KAFA POZİSYONU ---
        nose_tip = find_landmarks(1, frame, face_landmarks)
        chin_point = find_landmarks(152, frame, face_landmarks)
        face_center_x = (face_left[0] + face_right[0]) / 2
        head_yaw = (nose_tip[0] - face_center_x) / face_width
        head_pitch = euclidean_dist(nose_tip, chin_point) / face_width

        # --- 3. DUYGU ve STRES  ---
        
       
        lb_inner = find_landmarks(107, frame, face_landmarks)
        lb_outer = find_landmarks(70, frame, face_landmarks)
        rb_inner = find_landmarks(336, frame, face_landmarks)
        rb_outer = find_landmarks(300, frame, face_landmarks)
        left_brow_slope = (lb_inner[1] - lb_outer[1]) / face_width
        right_brow_slope = (rb_inner[1] - rb_outer[1]) / face_width
        avg_brow_slope = (left_brow_slope + right_brow_slope) / 2

        
        brow_dist_norm = euclidean_dist(lb_inner, rb_inner) / face_width
        
      
        lip_line_h = euclidean_dist(find_landmarks(78, frame, face_landmarks), find_landmarks(308, frame, face_landmarks))
        lip_line_v = euclidean_dist(find_landmarks(13, frame, face_landmarks), find_landmarks(14, frame, face_landmarks))
        lip_ratio_norm = (lip_line_h / (lip_line_v + 1e-6)) / face_width 

        
        brow_pt = find_landmarks(65, frame, face_landmarks)
        eye_pt = find_landmarks(159, frame, face_landmarks)
        brow_lift_norm = euclidean_dist(brow_pt, eye_pt) / face_width 

        mouth_left = find_landmarks(61, frame, face_landmarks)
        mouth_right = find_landmarks(291, frame, face_landmarks)
        mouth_width_norm = euclidean_dist(mouth_left, mouth_right) / face_width 

        mouth_top = find_landmarks(13, frame, face_landmarks)
        mouth_bot = find_landmarks(14, frame, face_landmarks)
        mouth_open_norm = euclidean_dist(mouth_top, mouth_bot) / face_width 

        lip_center = find_landmarks(0, frame, face_landmarks)
        mouth_corner_avg_y = (mouth_left[1] + mouth_right[1]) / 2
        frown_score_norm = (mouth_corner_avg_y - lip_center[1]) / face_width

        ear_l = calculate_ear(face_landmarks, frame, is_right=False)
        ear_r = calculate_ear(face_landmarks, frame, is_right=True)
        ear_mean = (ear_l + ear_r) / 2

        # ==========================
        #   KALİBRASYON MODU
        # ==========================
        if not user_exists:
            if calib_start_time is None: calib_start_time = time.time()
            elapsed = time.time() - calib_start_time
            cv2.putText(frame, f"KALIBRASYON: Kameraya Duz ve Notr Bak ({3-int(elapsed)})", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            collecting_data.append([brow_lift_norm, mouth_width_norm, mouth_open_norm, avg_brow_slope, frown_score_norm, ear_mean, head_pitch, brow_dist_norm, lip_ratio_norm])

            if elapsed > 3:
                means = np.mean(np.array(collecting_data), axis=0)
                calib = {
                    "brow_lift_base": float(means[0]),
                    "mouth_width_base": float(means[1]),
                    "mouth_open_base": float(means[2]),
                    "brow_slope_base": float(means[3]), 
                    "frown_base": float(means[4]),
                    "ear_base": float(means[5]),
                    "head_pitch_base": float(means[6]),
                    "brow_dist_base": float(means[7]),
                    "lip_ratio_base": float(means[8])
                }
                db["users"][user_name] = calib
                save_all_users(db)
                user_exists = True
                print("Kalibrasyon Tamamlandı ve Kaydedildi.")
            
            cv2.imshow('Sistem', frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        # ==========================
        #   ANALİZ VE ARAYÜZ
        # ==========================
        
       
        d_lift = brow_lift_norm - calib["brow_lift_base"]
        d_width = mouth_width_norm - calib["mouth_width_base"]
        d_open = mouth_open_norm - calib["mouth_open_base"]
        d_slope = avg_brow_slope - calib["brow_slope_base"]
        d_frown = frown_score_norm - calib["frown_base"]
        d_brow_dist = brow_dist_norm - calib["brow_dist_base"]
        d_lip_ratio = lip_ratio_norm - calib["lip_ratio_base"]


        # --- KAFA POZİSYONU---
        pose_text = "ONUNE BAKIYO"
        pose_color = (0, 255, 0)
        
        if abs(head_yaw) > 0.15: 
            pose_text = "SAGA/SOLA BAKIYO"
            pose_color = (0, 165, 255)
        
        if head_pitch < calib["head_pitch_base"] - 0.10:
            pose_text = "BASINI KALDIR"
            pose_color = (0, 0, 255)

        cv2.putText(frame, f"Bas Durumu: {pose_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pose_color, 2)


        
        ear_threshold = calib["ear_base"] * 0.75 
        instant_sleep = 100 if ear_mean < ear_threshold else 0 

      
        
        sleep_pct = sleep_pct * (1 - SLEEP_SMOOTHING_FACTOR) + instant_sleep * SLEEP_SMOOTHING_FACTOR

        
        bar_x, bar_y, bar_w, bar_h = 20, 70, 200, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), 2)
        
        fill_w = int((sleep_pct / 100) * bar_w)
        bar_color = (0, 255, 0) 
        if sleep_pct > 20: bar_color = (0, 255, 255) 
        if sleep_pct > 40: bar_color = (0, 0, 255)   
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, f"UYKU: %{int(sleep_pct)}", (bar_x + bar_w + 10, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
        

        # --- STRES TESPİTİ  ---
        
        instant_stress_score = 0.0
        
         
        if d_brow_dist < -0.015: # Kaşlar yaklaştı
            instant_stress_score = max(instant_stress_score, 0.5) 
        if d_lip_ratio < -0.015: # Dudaklar gergin/basık
            instant_stress_score = max(instant_stress_score, 0.5) 

       
        stress_pct = stress_pct * (1 - STRESS_SMOOTHING_FACTOR) + (instant_stress_score * 100) * STRESS_SMOOTHING_FACTOR

        stress_bar_y = 100
        cv2.rectangle(frame, (bar_x, stress_bar_y), (bar_x + bar_w, stress_bar_y + bar_h), (50, 50, 50), 2)
        
        stress_fill_w = int((stress_pct / 100) * bar_w)
        stress_bar_color = (255, 255, 0) 
        if stress_pct > 30: stress_bar_color = (0, 165, 255) 
        if stress_pct > 60: stress_bar_color = (0, 0, 255) 
        
        cv2.rectangle(frame, (bar_x, stress_bar_y), (bar_x + stress_fill_w, stress_bar_y + bar_h), stress_bar_color, -1)
        cv2.putText(frame, f"STRES: %{int(stress_pct)}", (bar_x + bar_w + 10, stress_bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stress_bar_color, 2)


        # --- DUYGU TESPİTİ  ---
        
        emotion = "Notr"
        emo_color = (255, 255, 255)

      
        if d_slope > 0.025: 
            emotion = "KIZGIN" 
            emo_color = (0, 0, 255)
            
        
        elif d_width > 0.03 and d_frown < 0:
            emotion = "MUTLU"
            emo_color = (0, 255, 255)
        elif d_open > 0.05:
            emotion = "ESNEME"
            emo_color = (255, 0, 255)
        elif d_frown > 0.015:
            emotion = "UZGUN"
            emo_color = (255, 0, 0)
        
        cv2.putText(frame, f"Duygu: {emotion}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, emo_color, 2)


    
        if sleep_pct > 50 or pose_text == "ONUNE BAK" or stress_pct > 75:
            cv2.putText(frame, "ACIL UYARI", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

    cv2.imshow('Sistem', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()