
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import mediapipe as mp
from collections import defaultdict
import math
import time

app = FastAPI()

# CORS - herkes erişebilsin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mediapipe Face Mesh (statik mod değil, ama tek kare işliyoruz)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)

# Göz ve nokta indeksleri (MediaPipe)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Global oturum istatistikleri: key -> {total_frames, closed_frames, continuous_closed_frames, last_closed_timestamp}
session_stats = defaultdict(lambda: {
    "total_frames": 0,
    "closed_frames": 0,
    "continuous_closed_frames": 0,
    "last_continuous_reset": time.time()
})

# Eşikler (ayarlayabilirsin)
EAR_THRESHOLD = 0.22          # göz kapalı sayılması için
CONTINUOUS_CLOSED_FRAMES_ALERT = 10  # kaç ardışık kapalı karede (sunucu kare/saniye'ne göre) alarm
# (frontend saniyede 1 kare gönderiyorsa 10 ~ 10 saniye)

def landmark_to_point(landmark, w, h):
    return np.array([int(landmark.x * w), int(landmark.y * h)])

def eye_ear(landmarks, idxs, w, h):
    p1 = landmark_to_point(landmarks[idxs[0]], w, h)
    p2 = landmark_to_point(landmarks[idxs[1]], w, h)
    p3 = landmark_to_point(landmarks[idxs[2]], w, h)
    p4 = landmark_to_point(landmarks[idxs[3]], w, h)
    p5 = landmark_to_point(landmarks[idxs[4]], w, h)
    p6 = landmark_to_point(landmarks[idxs[5]], w, h)
    # EAR = (|p2-p3| + |p4-p5|) / (2 * |p1-p6|)
    ear = (np.linalg.norm(p2 - p3) + np.linalg.norm(p4 - p5)) / (2.0 * (np.linalg.norm(p1 - p6) + 1e-6))
    return ear

@app.get("/")
def home():
    return {"status": "Backend çalışıyor!", "message": "FaceMesh analiz hazır."}

@app.post("/analyze")
async def analyze(request: Request, frame: UploadFile = File(...)):
    try:
        # Oturum anahtarı (daha iyi: frontend X-Client-Id header gönderir)
        client_host = request.client.host if request.client else "unknown"
        client_id = client_host

        # Görüntüyü okuma
        img_bytes = await frame.read()
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Görüntü alınamıyor veya bozuk."}, status_code=400)

        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Default dönüş
        if not results.multi_face_landmarks:
            # Oturumu yine say (frame sayısı artsın)
            session_stats[client_id]["total_frames"] += 1
            return JSONResponse({
                "error": "Yüz bulunamadı",
                "emotion": "Neutral",
                "stress": 0,
                "sleepiness": 0,
                "EAR": None,
                "PERCLOS": 0
            }, status_code=200)

        landmarks = results.multi_face_landmarks[0].landmark

        # EAR hesapla
        left_ear = eye_ear(landmarks, LEFT_EYE_IDX, w, h)
        right_ear = eye_ear(landmarks, RIGHT_EYE_IDX, w, h)
        EAR = (left_ear + right_ear) / 2.0

        # Basit yüz ölçüleri (normalize etmek için)
        left_face = landmark_to_point(landmarks[234], w, h)
        right_face = landmark_to_point(landmarks[454], w, h)
        face_width = np.linalg.norm(left_face - right_face) + 1e-6

        # Kaş ve göz arasındaki mesafe (brow lift proxy)
        brow_point = landmark_to_point(landmarks[65], w, h)
        eye_point = landmark_to_point(landmarks[159], w, h)
        brow_lift = np.linalg.norm(brow_point - eye_point)

        # Dudak genişliği (mutluluk proxy)
        mouth_left = landmark_to_point(landmarks[61], w, h)
        mouth_right = landmark_to_point(landmarks[291], w, h)
        mouth_width = np.linalg.norm(mouth_left - mouth_right)

        # Oturum istatistiklerini güncelle
        stats = session_stats[client_id]
        stats["total_frames"] += 1
        stats_touch = stats  # alias

        if EAR < EAR_THRESHOLD:
            stats_touch["closed_frames"] += 1
            stats_touch["continuous_closed_frames"] += 1
            stats_touch["last_continuous_reset"] = time.time()
        else:
            # continuous reset
            stats_touch["continuous_closed_frames"] = 0
            stats_touch["last_continuous_reset"] = time.time()

        # PERCLOS (basit): kapalı_frames / total_frames
        PERCLOS = stats_touch["closed_frames"] / max(1, stats_touch["total_frames"])
        # Convert to percentage 0-100
        sleepiness_pct = min(100, round(PERCLOS * 100))

        # continuous closed alarm (saniyeye bağlı değil; kare sayısına bağlı)
        continuous_alert_frames = stats_touch["continuous_closed_frames"]

        # Basit "stres" tahmini:
        # brow_lift normalised: yüksek değer -> şaşkınlık/gerginlik? (kaba tahmin)
        # Daha mantıklı: kaşların iç kısmının birbirine yakınlaşması stres/anger gösterebilir,
        # ama biz burada brow_lift/face_width oranını kullanalım.
        brow_norm = (brow_lift / face_width)  # yaklaşık oran
        # Normalleştir: 0.02..0.2 aralığını 0..100'e map et
        stress_score = int(np.clip((brow_norm - 0.03) * 800, 0, 100))  # ayarlanabilir katsayı

        # Duygu tahmini (basit kurallar; kesin model değil)
        # Öncelik: Surprise > Happy > Angry > Sad > Neutral
        emotion = "Neutral"
        # heuristics (ayarlanabilir thresholdlar)
        if brow_lift > (0.22 * face_width):  # çok yüksek kaş -> şaşkın
            emotion = "Surprise"
        elif mouth_width > (0.14 * face_width):  # geniş ağız -> mutlu (gülme)
            emotion = "Happy"
        else:
            # kaş içi (brow furrow) ölçümü için birkaç nokta:
            # kullanabileceğimiz proxy: distance between landmarks 10 (inner eyebrow) and 151 (nose root area)
            try:
                brow_inner_left = landmark_to_point(landmarks[70], w, h)
                brow_inner_right = landmark_to_point(landmarks[300], w, h)
                brow_inner_dist = np.linalg.norm(brow_inner_left - brow_inner_right)
            except Exception:
                brow_inner_dist = face_width * 0.25

            # normalize
            brow_inner_norm = brow_inner_dist / face_width

            # Sinirli (angry) tahmini: kaşların birbirine yakın + düşük brow_lift
            if brow_inner_norm < 0.12 and brow_lift < (0.08 * face_width):
                emotion = "Angry"
            # Üzgün (sad) tahmini: dar ağız + düşük brow_lift
            elif mouth_width < (0.10 * face_width) and brow_lift < (0.10 * face_width):
                emotion = "Sad"
            else:
                emotion = "Neutral"

        # Eğer çok uzun süredir ardışık kapalı ise, sleepiness'i daha yüksek göster
        if continuous_alert_frames >= CONTINUOUS_CLOSED_FRAMES_ALERT:
            # Artır (ör: en az %60)
            sleepiness_pct = max(sleepiness_pct, 60)

        # Döndür
        result = {
            "emotion": emotion,                       # "Happy","Sad","Angry","Surprise","Neutral"
            "stress": int(stress_score),              # 0-100
            "sleepiness": int(sleepiness_pct),        # 0-100
            "EAR": round(float(EAR), 4),
            "PERCLOS": round(float(PERCLOS), 4),
            "continuous_closed_frames": int(stats_touch["continuous_closed_frames"])
        }

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
