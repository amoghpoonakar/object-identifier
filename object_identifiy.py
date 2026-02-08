# ================== IMPORTS ==================
from ultralytics import YOLO
import cv2
import random
import pyttsx3
import speech_recognition as sr
import threading
import queue
import time
import webbrowser

# ================== VARIABLES ==================
# DEFINE VARIABLES
DETECT_ENABLED = True
EXCLUDE_PERSON = True
CONFIDENCE_THRESHOLD = 0.30

model = YOLO("yolov8m.pt")

# ================== CAMERA SETUP ==================
# SETUP AND USE CAMERA
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

print(
    "Camera Resolution:",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

window_name = "Advanced Vision System"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ================== COLORS ==================
# COLORS AND STUFF
random.seed(42)
CLASS_COLORS = {
    i: (random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255))
    for i in range(80)
}

CLASS_NAMES = model.names
ACTIVE_CLASSES = list(range(1, 80)) if EXCLUDE_PERSON else list(range(80))

# ================== TEXT-TO-SPEECH ==================
# SPEAK ABOUT OBJECT
VOICE_ENABLED = False
speech_queue = queue.Queue()

def tts_loop():
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

tts_thread = threading.Thread(target=tts_loop, daemon=True)
tts_thread.start()

def speak(text):
    if VOICE_ENABLED:
        speech_queue.put(text)

# ================== VOICE LISTENER ==================
# MICROPHONE INPUT
def listen_loop():
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)

    while True:
        if not VOICE_ENABLED:
            time.sleep(0.3)
            continue

        try:
            with mic as source:
                audio = r.listen(source, phrase_time_limit=6)

            command = r.recognize_google(audio).lower()
            print("Voice:", command)

            if "search photo of" in command:
                query = command.replace("search photo of", "").strip()
                if query:
                    speak(f"Searching images of {query}")
                    webbrowser.open(f"https://www.google.com/search?tbm=isch&q={query}")

        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            speak("Speech service unavailable")
        except Exception as e:
            print("Listener error:", e)
            time.sleep(1)

listener_thread = threading.Thread(target=listen_loop, daemon=True)
listener_thread.start()

# ================== MAIN LOOP ==================
# OBJECT DETECTION STARTS
spoken_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    output = frame.copy()

    # ================== DETECTION PAUSED ==================
    if not DETECT_ENABLED:
        cv2.putText(
            output,
            "DETECTION PAUSED (Press P)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        cv2.imshow(window_name, output)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            DETECT_ENABLED = True
        continue

    current_objects = set()
    object_count = {}

    results = model(
        frame,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        classes=ACTIVE_CLASSES,
        verbose=False
    )

    detections = results[0].boxes

    if detections is not None:
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = CLASS_NAMES[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS[cls]

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                f"{name} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            object_count[name] = object_count.get(name, 0) + 1

            if conf >= CONFIDENCE_THRESHOLD:
                current_objects.add(name)
                if VOICE_ENABLED and name not in spoken_objects:
                    speak(f"I can see a {name}")
                    spoken_objects.add(name)

    spoken_objects &= current_objects

    # ================== HUD ==================
    y = 30
    cv2.putText(
        output,
        "Objects Detected:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    for obj, count in object_count.items():
        y += 25
        cv2.putText(
            output,
            f"{obj}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    status = "Voice: ON (E to stop)" if VOICE_ENABLED else "Voice: OFF (A to start)"
    color = (0, 255, 0) if VOICE_ENABLED else (0, 0, 255)

    cv2.putText(
        output,
        status,
        (20, output.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.imshow(window_name, output)

    # ================== CAMERA WINDOW CONTROL ==================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('p'):
        DETECT_ENABLED = False

    if key == ord('a'):
        if not VOICE_ENABLED:
            VOICE_ENABLED = True
            speak("Voice assistant activated. Please speak.")
            print("Voice ON")

    if key == ord('e'):
        VOICE_ENABLED = False
        spoken_objects.clear()
        print("Voice OFF")

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)
