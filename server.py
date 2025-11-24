from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

# --------------------------------------------------
# BACKEND BAŞLANGIÇ
# --------------------------------------------------

app = FastAPI()

# 🔥 CORS — HERKES ERİŞEBİLSİN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # İstersen sadece kendi siteni yazabilirsin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 1) BACKEND STATUS KONTROLÜ (isteğe bağlı)
# --------------------------------------------------
@app.get("/")
def home():
    return {"status": "Backend çalışıyor!", "message": "PythonAnywhere FastAPI aktif."}


# --------------------------------------------------
# 2) ANALİZ ENDPOINTİ — KAMERA FRAME ALIR
# --------------------------------------------------
@app.post("/analyze")
async def analyze(frame: UploadFile = File(...)):
    try:
        # ---- FRAME OKUMA ----
        img_bytes = await frame.read()
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Görüntü alınamıyor"}, status_code=400)

        # ------------------------------------------------------
        # 🚧 ŞU ANLIK DEMO ANALİZ (Model eklemeye hazır alan)
        # ------------------------------------------------------
        # Buraya kendi model analiz algoritmanı ekleyebilirsin.
        # Şimdilik backend çalışsın diye örnek değer veriyorum:
        demo_emotion = "neutral"
        demo_stress = 27
        demo_sleepiness = 12

        result = {
            "emotion": demo_emotion,
            "stress": demo_stress,
            "sleepiness": demo_sleepiness
        }

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --------------------------------------------------
# BACKEND BİTİŞ
# --------------------------------------------------