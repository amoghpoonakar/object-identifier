# Object Identifier

---

## Project Overview

Object Identifier is a real-time computer vision system that detects and identifies objects using a deep learning model. It integrates voice assistance to enhance interactivity by providing spoken feedback and supporting voice commands.

---

## Features

- Real-time object detection using YOLOv8  
- High-resolution webcam support (1080p, 30 FPS)  
- Detection filtering using confidence thresholds  
- Option to exclude specific classes (e.g., person)  
- Interactive Heads-Up Display (HUD) showing detected objects and counts  
- Voice feedback for detected objects  
- Voice command support for image search  
- Keyboard controls for system interaction  

---

## Technology Stack

- Python  
- OpenCV  
- Ultralytics YOLOv8  
- Pyttsx3 (Text-to-Speech)  
- SpeechRecognition  
- Threading and Queue  

---

## How It Works

1. The webcam captures live video frames.  
2. YOLOv8 processes each frame to detect objects.  
3. Detected objects are displayed with bounding boxes and confidence scores.  
4. A HUD shows object counts in real time.  
5. When voice is enabled:
   - The system announces newly detected objects.  
   - It listens for voice commands to perform actions such as image search.  

---

## Controls

| Key | Action |
|-----|--------|
| Q   | Quit application |
| P   | Pause/Resume detection |
| A   | Enable voice assistant |
| E   | Disable voice assistant |

---

## Voice Commands

- "search photo of <object>"  
  Opens a web browser to display image search results for the specified object.

Example:
search photo of car

---

## Configuration

You can modify the following variables in the code:

- DETECT_ENABLED – Enable or disable detection at startup  
- EXCLUDE_PERSON – Exclude person class from detection  
- CONFIDENCE_THRESHOLD – Minimum confidence for voice announcements  
- VOICE_ENABLED – Enable or disable voice assistant  

---

## Output

- Live video feed with bounding boxes  
- Object labels with confidence scores  
- Real-time object count display  
- Voice announcements (optional)  

---

## Use Cases

- Smart surveillance systems  
- Assistive technology for visually impaired users  
- AI-based interactive systems  
- Educational demonstrations in computer vision  

---

## Limitations

- Requires a working webcam and microphone  
- Performance depends on system hardware  
- Speech recognition requires internet connectivity  
- Lower confidence thresholds may result in false detections  

---

## Future Improvements

- Support for custom-trained models  
- Enhanced voice command system  
- Graphical user interface (GUI)  
- Performance optimization for low-end systems  
