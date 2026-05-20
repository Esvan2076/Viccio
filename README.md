<p align="center">
  <h1 align="center">Viccio</h1>
  <p align="center">
    <strong>Intelligent voice-controlled assistive wearable for visually impaired users</strong>
  </p>
  <p align="center">
    Real-time object detection &bull; Sign &amp; text reading &bull; 100% local AI &bull; Hands-free voice control
  </p>
  <p align="center">
    <em>Built at GuadalaHacks 2026</em>
  </p>
</p>

---

## Table of Contents

- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [System Architecture](#system-architecture)
- [How We Built It](#how-we-built-it)
  - [Hardware — ESP32-CAM Smart Glasses](#hardware--esp32-cam-smart-glasses)
  - [Android Application](#android-application)
  - [AI / Computer Vision Pipeline](#ai--computer-vision-pipeline)
  - [Voice Interface](#voice-interface)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Application Modes](#application-modes)
- [Voice Commands](#voice-commands)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [ESP32-CAM Setup](#esp32-cam-setup)
- [Challenges We Ran Into](#challenges-we-ran-into)
- [Accomplishments We're Proud Of](#accomplishments-were-proud-of)
- [What We Learned](#what-we-learned)
- [What's Next for Viccio](#whats-next-for-viccio)
- [License](#license)

---

## Inspiration

To support a vulnerable population through a simple device that, unlike other technologies, **does not inhibit their senses**. Our goal is to empower users through an intuitive, non-intrusive solution that grants greater independence to visually impaired individuals in everyday environments.

---

## What It Does

Viccio is an **intelligent, voice-controlled assistive wearable** that helps visually impaired users navigate their surroundings independently. By combining a camera-equipped pair of glasses with a smartphone app, the system processes visual data **100% locally** to assist the user in real time.

The system operates in dedicated modes — each optimized for a specific real-world task:

| Mode | Purpose | AI Engine |
|------|---------|-----------|
| **Reading Mode** | Reads emergency signs, exits, trash labels, and navigational text aloud | Google ML Kit OCR |
| **Restroom Mode** | Detects and announces restroom signage in the user's path | Custom YOLO11 model via TFLite |

All interaction is entirely **hands-free** through natural voice commands in **Mexican Spanish**, with full text-to-speech feedback.

---

## System Architecture

```
┌─────────────────────┐       MJPEG over HTTP        ┌──────────────────────────────┐
│   ESP32-CAM Module  │ ──────────────────────────▶   │    Android Application       │
│   (Smart Glasses)   │   Wi-Fi (local network)       │                              │
└─────────────────────┘                                │  ┌────────────────────────┐  │
                                                       │  │   Stream Executor      │  │
                                                       │  │   (Network I/O thread) │  │
                                                       │  └──────────┬─────────────┘  │
                                                       │             │ Bitmap frames  │
                                                       │             ▼                │
                                                       │  ┌────────────────────────┐  │
                                                       │  │   AI Executor          │  │
                                                       │  │   (Inference thread)   │  │
                                                       │  │                        │  │
                                                       │  │  ┌──────────────────┐  │  │
                                                       │  │  │ ML Kit OCR       │  │  │
                                                       │  │  │ (Reading Mode)   │  │  │
                                                       │  │  └──────────────────┘  │  │
                                                       │  │  ┌──────────────────┐  │  │
                                                       │  │  │ YOLO11 TFLite    │  │  │
                                                       │  │  │ (Restroom Mode)  │  │  │
                                                       │  │  └──────────────────┘  │  │
                                                       │  └──────────┬─────────────┘  │
                                                       │             │ Results        │
                                                       │             ▼                │
                                                       │  ┌────────────────────────┐  │
                                                       │  │   TTS + Overlay View   │  │
                                                       │  │   (User feedback)      │  │
                                                       │  └────────────────────────┘  │
                                                       │                              │
                                                       │  ┌────────────────────────┐  │
                                                       │  │   SpeechRecognizer     │  │
                                                       │  │   (Voice commands)     │  │
                                                       │  └────────────────────────┘  │
                                                       └──────────────────────────────┘
```

---

## How We Built It

### Hardware — ESP32-CAM Smart Glasses

A lightweight pair of glasses is equipped with an **ESP32-CAM** module configured to stream real-time MJPEG video over HTTP to the user's smartphone on a **localized, low-latency Wi-Fi connection**. The stream endpoint serves individual JPEG frames that are parsed byte-by-byte by the Android app, detecting JPEG SOI (`0xFFD8`) and EOI (`0xFFD9`) markers to extract each frame without relying on a third-party MJPEG library.

### Android Application

The control center is a **native Android application built in Java** (min SDK 24, target SDK 36). It acts as a central **state machine** managing three application modes — *System Active*, *Reading Mode*, and *Restroom Mode* — seamlessly transitioning between them depending on user behavior and real-time environment requirements.

Key architectural decisions:

- **Dual-executor threading model** — A dedicated `streamExecutor` handles network I/O (reading the MJPEG stream) while a separate `aiExecutor` runs inference, preventing AI workloads from blocking frame reception and preserving a smooth frame rate.
- **Foreground Service (`ViccioService`)** — Enables background processing with a persistent notification so the AI pipeline continues running even when the screen is off, maximizing battery-conscious independence.
- **Frame rate throttling** — A configurable `FRAME_INTERVAL` (150 ms) caps inference frequency to balance responsiveness with CPU/thermal constraints on mobile hardware.
- **Bitmap lifecycle management** — Careful `.recycle()` calls on cropped and rotated bitmaps prevent memory leaks during continuous frame processing.

### AI / Computer Vision Pipeline

#### Object Detection — YOLO11 + TensorFlow Lite

We trained a **custom object detection model using YOLO11** to recognize key environmental elements (restroom signage). To achieve 100% local execution, the model was exported to **TensorFlow Lite float32** format (`best_float32_500.tflite`, ~10 MB) and integrated directly into the Android app.

The `YoloDetector` class manages the full inference lifecycle:

| Step | Detail |
|------|--------|
| **Preprocessing** | Resize input bitmap to 320×320, normalize pixel values from `[0, 255]` to `[0.0, 1.0]` |
| **Inference** | Run the TFLite interpreter with 4 CPU threads on the output tensor of shape `[1, 5, 8400]` |
| **Post-processing** | Extract bounding boxes (`cx, cy, w, h` → corner format), apply a 50% confidence threshold |
| **NMS** | Greedy Non-Maximum Suppression with IoU threshold of 0.45 to eliminate duplicate detections |

Detections above **60% confidence** trigger a Spanish TTS announcement (e.g., *"Hay un baño de frente"*), with a 3-second cooldown to prevent repetitive alerts.

#### Text Recognition — Google ML Kit OCR

For reading emergency and navigational signs, we integrated **Google ML Kit's on-device text recognition**. A custom image processing pipeline **crops the camera feed to a central ~50% bounding region** (70.7% width × 70.7% height), forcing the system to focus on what the user is pointing at and filtering out peripheral noise.

Recognized text is run through a **regex-based sign classifier** that identifies:

- Emergency exits (*"Salida de emergencia"*)
- Fire extinguishers (*"Extintor"*)
- Exits (*"Salida"*)
- Alarms (*"Alarma"*)
- Evacuation routes (*"Ruta de evacuación"*)
- Waste sorting signs — organic, inorganic, plastic, metal

The regex patterns are intentionally **OCR-tolerant**, accounting for common misreads (e.g., `0` for `O`, `1` for `I`, `4` for `A`) to maximize reliability under imperfect real-world conditions.

A **stability mechanism** requires the same sign to be detected across 2 consecutive frames before announcing, with a 1.5-second cooldown to avoid repeated speech.

### Voice Interface

Navigation is entirely hands-free via Android's native **`SpeechRecognizer`** configured for **Mexican Spanish (`es-MX`)**. The speech recognizer runs continuously in the background, automatically restarting 700 ms after each utterance ends or on error, creating a seamless always-listening experience.

**Text-to-Speech** output uses the `es-MX` locale and includes an `UtteranceProgressListener` that tracks speaking state — the system automatically **suppresses incoming voice commands while it is speaking** to prevent the device from interpreting its own audio output as user input.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Java 11 |
| **Platform** | Android (min SDK 24 / Android 7.0+) |
| **Build System** | Gradle (Kotlin DSL) |
| **UI** | ConstraintLayout, custom `OverlayView` (Canvas-based) |
| **Camera Stream** | ESP32-CAM MJPEG over HTTP (custom byte-level parser) |
| **Object Detection** | YOLO11 → TensorFlow Lite (float32, 320×320 input) |
| **Text Recognition** | Google ML Kit (on-device, latin script) |
| **Voice Input** | Android SpeechRecognizer (offline, `es-MX`) |
| **Voice Output** | Android TextToSpeech (`es-MX`) |
| **Hardware** | ESP32-CAM module mounted on lightweight glasses |

---

## Project Structure

```
Viccio/
├── app/
│   ├── build.gradle.kts              # Dependencies: CameraX, TFLite, ML Kit, Material
│   └── src/main/
│       ├── AndroidManifest.xml        # Permissions: INTERNET, RECORD_AUDIO
│       ├── assets/
│       │   ├── best_float32_500.tflite   # Custom YOLO11 model (~10 MB)
│       │   └── labels.txt                # Detection class labels
│       ├── java/com/example/myapplication/
│       │   ├── MainActivity.java         # Main UI, stream display, orchestration
│       │   ├── ViccioService.java        # Foreground service for background processing
│       │   ├── YoloDetector.java         # TFLite inference engine with NMS
│       │   ├── DetectionResult.java      # Data model for detection outputs
│       │   └── OverlayView.java          # Canvas-based bounding box renderer
│       └── res/
│           └── layout/
│               └── activity_main.xml     # UI: stream view, overlay, status labels
├── build.gradle.kts
├── settings.gradle.kts
└── README.md
```

---

## Application Modes

The app functions as a **finite state machine** with the following states:

```
                  "INICIAR"
    ┌──────────┐ ─────────▶ ┌──────────────┐
    │  IDLE    │             │ SYSTEM ACTIVE │
    └──────────┘ ◀───────── └──────┬───────┘
                  "APAGAR"         │
                           ┌───────┴───────┐
                   "LEER"  │               │  "BAÑO"
                     ▼                           ▼
              ┌──────────────┐         ┌──────────────────┐
              │ READING MODE │         │  RESTROOM MODE   │
              │  (ML Kit OCR)│         │ (YOLO11 TFLite)  │
              └──────────────┘         └──────────────────┘
```

| State | Description |
|-------|-------------|
| **Idle** | Streaming video but not processing. Listening for wake command. |
| **System Active** | Core system is on. Awaiting mode selection. |
| **Reading Mode** | OCR pipeline active — reads signs and text aloud. |
| **Restroom Mode** | YOLO pipeline active — detects and announces restrooms. |

---

## Voice Commands

| Command (es-MX) | Alternative | Action |
|------------------|-------------|--------|
| *"Iniciar"* | *"Start"* | Activates the system |
| *"Leer"* / *"Texto"* | *"Sign"* | Enters Reading Mode (OCR) |
| *"Baño"* / *"Baños"* | *"Restroom"* | Enters Restroom Mode (YOLO) |
| *"Apagar"* / *"Detener"* | *"Stop"* | Shuts down all processing |

---

## Getting Started

### Prerequisites

- **Android Studio** Hedgehog (2023.1) or later
- **Android SDK 36** with build tools
- **ESP32-CAM** module (AI-Thinker or compatible) flashed with MJPEG streaming firmware
- An Android device running **Android 7.0 (API 24)** or higher

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Esvan2076/Viccio.git
   cd Viccio
   ```

2. **Open in Android Studio**
   - File → Open → select the `Viccio/` root directory
   - Let Gradle sync complete

3. **Configure the ESP32 stream URL**
   - Open `MainActivity.java` (line 50) and/or `ViccioService.java` (line 48)
   - Update `STREAM_URL` to match your ESP32-CAM's IP address:
     ```java
     private static final String STREAM_URL = "http://<YOUR_ESP32_IP>:81/stream";
     ```

4. **Build and deploy**
   - Connect your Android device via USB
   - Click **Run ▶** in Android Studio

### ESP32-CAM Setup

1. Flash your ESP32-CAM with the standard **CameraWebServer** example (available in Arduino IDE under *File → Examples → ESP32 → Camera → CameraWebServer*)
2. Connect the ESP32-CAM to the same Wi-Fi network as the Android device
3. Note the IP address printed to the serial monitor — use this for `STREAM_URL`

---

## Challenges We Ran Into

| Challenge | Details |
|-----------|---------|
| **Java ↔ Model Compatibility** | Ensuring our AI model integrated smoothly with native Android Java code and the TFLite runtime. |
| **Hardware-Software Sync** | Establishing a stable, low-latency communication link between the ESP32 hardware and the smartphone application, including a custom byte-level MJPEG parser. |
| **Model Optimization** | Training, refining, and compressing a computer vision model to run efficiently on mobile hardware within a ~10 MB footprint. |
| **Edge AI Deployment** | Implementing local model inference on a smartphone without relying on cloud servers, while maintaining acceptable latency (<150 ms per frame). |
| **Data Stream Quality** | Balancing high video quality with a consistent frame rate while streaming data from the ESP32 over Wi-Fi. |
| **Voice Command Management** | Handling continuous background voice recognition without picking up audio feedback from the device's own speakers — solved with an `UtteranceProgressListener`-based suppression mechanism. |

---

## Accomplishments We're Proud Of

- **Social Impact** — Building a viable, high-quality solution engineered specifically to empower a vulnerable population.
- **Model Precision** — Achieving high prototype accuracy and reliable detection despite working under limited hardware and budget constraints.
- **100% Local Execution** — Keeping all data processing on the device to guarantee absolute user privacy and offline functionality. Zero cloud dependencies.
- **Dual-threaded Architecture** — Separating network I/O from AI inference for smooth real-time performance on consumer-grade smartphones.

---

## What We Learned

- How to train advanced computer vision models using **YOLO11** and optimize them for local edge deployment via TensorFlow Lite.
- Mastering the integration of **Java and TensorFlow Lite** inside Android Studio for real-time inference with custom post-processing (NMS, confidence thresholding).
- Designing robust software architectures that handle **sensor fusion** (camera + microphone) efficiently on Android with proper thread isolation.
- Building **OCR-tolerant regex classifiers** that account for real-world text recognition imperfections.

---

## What's Next for Viccio

| Goal | Description |
|------|-------------|
| **Expand Detection** | Train the model to recognize a wider variety of everyday objects and potential hazards. |
| **UX Refinement** | Continuously improve the user experience based on direct feedback from visually impaired users. |
| **Cost Optimization** | Increase video and processing quality without driving up hardware and manufacturing costs. |
| **Mass Production** | Create a scalable manufacturing and hardware assembly strategy. |
| **Strategic Partnerships** | Forge alliances with NGOs, health organizations, and strategic allies to scale growth and reach those who need it most. |



---

<p align="center">
  <strong>Viccio</strong> — Empowering independence through local AI.
</p>
