import io
import os
import threading
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from disease_medicine import disease_medicine_map

# Optional TTS
try:
    import pyttsx3
    _has_pyttsx3 = True
except Exception:
    _has_pyttsx3 = False

try:
    from gtts import gTTS
    from playsound import playsound
    _has_gtts = True
except Exception:
    _has_gtts = False

# Config
MODEL_PATH = "best.pt"
OUTPUT_PATH = os.path.join("static", "output.jpg")
ALLOWED_EXT = {"png", "jpg", "jpeg"}

# Ensure static exists
os.makedirs("static", exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load model (this may take time)
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def speak_text(text, lang_code="en"):
    """
    Try pyttsx3 first (offline). If not available, try gTTS (requires internet).
    Play sound in a background thread so it doesn't block the Flask response.
    """
    def _play_pyttsx3(txt, lang):
        try:
            engine = pyttsx3.init()
            # try to set voice based on language tag heuristically
            voices = engine.getProperty('voices')
            chosen = None
            lang_lower = lang.lower()
            for v in voices:
                v_name = str(v.name).lower()
                v_id = str(v.id).lower()
                if "hindi" in v_name or "hi" in v_id or "hindi" in v_id:
                    chosen = v.id
                    break
                if "kannada" in v_name or "kannada" in v_id:
                    chosen = v.id
                    break
            if chosen:
                engine.setProperty('voice', chosen)
            engine.say(txt)
            engine.runAndWait()
        except Exception as e:
            print("pyttsx3 speak failed:", e)

    def _play_gtts(txt, lang):
        try:
            tts = gTTS(txt, lang=lang)
            tmp = "static/tts_temp.mp3"
            tts.save(tmp)
            playsound(tmp)
            try:
                os.remove(tmp)
            except: pass
        except Exception as e:
            print("gTTS failed:", e)

    if _has_pyttsx3:
        t = threading.Thread(target=_play_pyttsx3, args=(text, lang_code))
        t.daemon = True
        t.start()
        return True
    elif _has_gtts:
        # map lang_code to gTTS codes
        code = "en"
        if lang_code.startswith("kn") or lang_code == "kn":
            code = "kn"  # gTTS may not support Kannada; will fallback
        elif lang_code.startswith("hi") or lang_code == "hi":
            code = "hi"
        t = threading.Thread(target=_play_gtts, args=(text, code))
        t.daemon = True
        t.start()
        return True
    else:
        print("No TTS available (pyttsx3/gTTS). Skipping speech.")
        return False

def draw_neon_boxes(result, img):
    """
    result: a single Results object (from model(img))
    img: numpy BGR image (OpenCV)
    returns: annotated image (BGR)
    """
    overlay = img.copy().astype(np.uint8)
    h, w = img.shape[:2]

    boxes = []
    if result.boxes is not None:
        # xyxy, scores, cls
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else [1.0]*len(xyxy)
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls in zip(xyxy, confs, clss):
            boxes.append((box, float(conf), int(cls)))

    # Neon style: draw multiple rectangles with increasing thickness and decreasing alpha
    for box, conf, cls in boxes:
        x1, y1, x2, y2 = map(int, box)
        # class name lookup
        cls_name = result.names.get(cls, str(cls))
        # color pick based on cls
        color = (0, 255, 255)  # base cyan-ish
        # Draw glow by repeated rectangles
        max_glow = 6
        for i in range(max_glow, 0, -1):
            alpha = 0.02 * i
            thickness = int((max_glow - i) * 2 + 1)
            # draw on overlay
            cv2.rectangle(overlay, (x1 - i, y1 - i), (x2 + i, y2 + i), color, thickness=thickness, lineType=cv2.LINE_AA)
        # Put label background
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, y1 - th - 14), (x1 + tw + 8, y1), (0,0,0), -1)
        cv2.putText(overlay, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    # blend overlay with original for neon translucent look
    annotated = cv2.addWeighted(overlay, 0.9, img, 0.1, 0)
    return annotated

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    # lang param: "en", "kn", "hi"
    lang = request.form.get("lang", "en")
    voice = request.form.get("voice", "false").lower() == "true"

    # accept uploaded file
    if "image" not in request.files:
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        return "Unsupported file type", 400

    # read image bytes -> OpenCV BGR
    in_memory = file.read()
    pil = Image.open(io.BytesIO(in_memory)).convert("RGB")
    img = np.array(pil)[:, :, ::-1].copy()  # RGB->BGR

    # Run model (single image)
    results = model(pil)  # pass PIL directly
    result = results[0]

    # Annotate image
    annotated = draw_neon_boxes(result, img.copy())
    # Save annotated image
    cv2.imwrite(OUTPUT_PATH, annotated)

    # Build response: list of detections with medicines
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        xy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else [1.0]*len(xy)
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_idx in zip(xy, confs, clss):
            name = result.names.get(int(cls_idx), str(cls_idx))
            # Lookup medicine
            medicine = None
            if name in disease_medicine_map:
                if lang == "kn":
                    medicine = disease_medicine_map[name].get("kn", disease_medicine_map[name].get("en"))
                elif lang == "hi":
                    medicine = disease_medicine_map[name].get("hi", disease_medicine_map[name].get("en"))
                else:
                    medicine = disease_medicine_map[name].get("en")
            else:
                medicine = "⚠️ Treatment info coming soon."

            detections.append({
                "class": name,
                "confidence": float(conf),
                "medicine": medicine
            })
    else:
        # No boxes
        detections.append({
            "class": "No detection",
            "confidence": 0.0,
            "medicine": {
                "en": "No disease detected.",
                "kn": "ಯಾವುದೇ ರೋಗ ಕಂಡುಬಂದಿಲ್ಲ.",
                "hi": "कोई रोग नहीं मिला।"
            }.get(lang, "No disease detected.")
        })

    # Voice speak first detection's medicine if requested
    if voice and len(detections) > 0:
        first_med = detections[0]["medicine"]
        # choose language code for TTS
        lang_code = "en"
        if lang == "kn":
            lang_code = "kn"
        elif lang == "hi":
            lang_code = "hi"
        speak_text(first_med, lang_code=lang_code)

    # Return template showing image and detection cards
    return render_template("index.html", output_image=url_for('static', filename='output.jpg'), detections=detections, lang=lang)

if __name__ == "__main__":
    # run Flask
    app.run(host="0.0.0.0", port=7860, debug=True)
