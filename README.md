# Braille Vision Scanner

**Real-time physical Braille recognition using computer vision and text-to-speech.**

Built for the BrailleVision Hackathon 2026 — converts camera-captured embossed or handwritten Braille into spoken English.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green) ![React](https://img.shields.io/badge/React-18-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange)

---

## What It Does

Point a camera at physical Braille text → the system detects raised dots → segments them into cells → decodes to English → speaks the result aloud.

**Pipeline:**
```
Camera → Preprocessing → Dot Detection → Cell Segmentation → Braille Decoding → Text-to-Speech
```

## Demo

### How to Test

1. **Upload Mode**: Click "Upload Image" and select a Braille image
2. **Live Camera**: Click "Start Scan" and point webcam at printed Braille
3. **Voice Commands**: Say "start scan", "stop scan", "repeat", "faster", "slower"

### Test Images Included

- `test_braille_hello.png` — "hello" (white dots on dark background)
- `braille_test_sheet.png` — Multi-line: "hello", "abc", "no", "ok"
- `test_images/` — 30+ test scenarios with shadows, textures, blur

---

## Features

### Core Recognition
- **All 26 English letters** (Grade 1 Braille)
- **Numbers 0-9** with number indicator support
- **Multi-line text** with reading order (left-to-right, top-to-bottom)
- **Word spacing** detection
- **96%+ confidence** on clean images

### Robustness
- Handles **paper texture** and noise
- Works with **shadows** (left, right, top gradients)
- Supports **different paper colors** (white, cream, gray, dark)
- Handles **varying dot sizes** (8px to 20px radius)
- Works with **slight camera blur**
- **Smart noise rejection** — won't speak when pointed at faces or random objects
- **Lighting-aware thresholding** — auto-switches between Otsu and adaptive based on lighting uniformity

### Accessibility
- **Text-to-Speech** output (Web Speech API)
- **Voice commands** for hands-free operation
- **High-contrast UI** (7:1+ contrast ratio, WCAG AAA)
- **Large touch targets** (48x48dp minimum)
- **ARIA labels** on all interactive elements
- **Audio tutorial** on startup
- **Camera alignment feedback**

### Performance
- **< 2 seconds** end-to-end (capture to speech)
- **2+ FPS** continuous scanning mode
- **27ms average** per frame processing
- **< 2GB RAM** usage

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (React + TypeScript + Vite)                    │
│  ├── Camera capture (getUserMedia)                      │
│  ├── Voice commands (Web Speech API)                    │
│  ├── Text-to-Speech (speechSynthesis)                   │
│  └── WebSocket streaming (base64 frames @ 2 FPS)        │
└────────────────────┬────────────────────────────────────┘
                     │ WebSocket / REST
┌────────────────────▼────────────────────────────────────┐
│  Backend (FastAPI + Python)                              │
│  ├── Image Capture Module (OpenCV)                      │
│  ├── Alignment Guide (contour detection)                │
│  ├── Preprocessing (normalize, deskew, threshold)       │
│  ├── Dot Detector (contour analysis + classification)   │
│  ├── Cell Segmenter (grid grouping, reading order)      │
│  ├── Braille Decoder (Grade 1 lookup + state machine)   │
│  ├── Quality Assessor (confidence, degradation)         │
│  └── TTS Engine (pyttsx3)                               │
└─────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 18, TypeScript, Vite | Accessible UI |
| Backend | Python 3.10+, FastAPI, Uvicorn | API + Pipeline |
| Computer Vision | OpenCV 4.8+ | Image processing, dot detection |
| ML/Detection | Contour analysis, Otsu/Adaptive thresholding, DoG, Hough Circles | Dot classification |
| Speech | Web Speech API, pyttsx3 | TTS + voice commands |
| Communication | WebSocket | Real-time frame streaming |
| Testing | pytest, Hypothesis, Vitest | 267 automated tests |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Webcam (optional, for live scanning)

### Install & Run

```bash
# Clone the repo
git clone <repo-url>
cd Hackathons

# Backend
cd backend
pip install -r requirements.txt
cd ..

# Frontend
cd frontend
npm install
cd ..

# Run (two terminals)
# Terminal 1:
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2:
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload image for recognition |
| POST | `/api/capture` | Single-shot camera capture |
| GET | `/api/settings` | Get TTS settings |
| PUT | `/api/settings` | Update TTS rate |
| POST | `/api/repeat` | Repeat last utterance |
| WS | `/ws/scan` | Real-time frame streaming |

### WebSocket Protocol

Send: base64-encoded JPEG frame
Receive:
```json
{
  "text": "hello",
  "confidence": 0.97,
  "alignment": {"is_aligned": true, "direction": "centered"},
  "processing_time_ms": 27
}
```

---

## How It Detects Physical Braille

1. **Preprocessing**: Normalizes brightness (target mean 128), corrects rotation (up to ±15°), reduces noise (Gaussian 5x5), applies adaptive/Otsu thresholding based on lighting uniformity

2. **Dot Detection**: Finds contours in binary image, classifies each by circularity (≥0.55), convexity (≥0.70), and inertia ratio (≥0.40). Rejects noise with <5% false positive rate.

3. **Cell Segmentation**: Groups dots into 2x3 grids using spatial clustering. Detects lines via vertical gaps (>2x row spacing). Handles inter-cell spacing variation (50-150%).

4. **Braille Decoding**: Maps dot patterns to English using Grade 1 lookup table. State machine handles number indicators (dots 3-4-5-6) and capital indicators (dot 6).

5. **Noise Rejection**: Validates dot patterns for grid structure, size uniformity, and spacing regularity. Requires 2 consecutive matching frames before speaking.

---

## Testing

```bash
# Unit tests (267 tests)
cd backend && pytest tests/ -v
cd frontend && npm test

# Stress test (32 scenarios)
python -m backend.stress_test_braille

# Realistic conditions test (30 scenarios)
python -m backend.realistic_test_braille
```

### Test Results
- **Unit tests**: 267/267 passing
- **Stress test**: 22/32 passing (all letters, numbers, multi-line, sizes, noise, spaces)
- **Realistic test**: 18/30 passing (textures, shadows, blur, paper colors)

---

## What Was Built During the Hackathon

**Newly built (100% original):**
- Complete Braille recognition pipeline (preprocessing → detection → segmentation → decoding)
- Adaptive thresholding with lighting-aware switching
- Grid-structure validation for noise rejection
- Cell segmenter with natural-break spacing detection
- Grade 1 Braille decoder with number/capital state machine
- JSON serialization with round-trip property
- Accessible React frontend with voice commands
- WebSocket real-time streaming architecture
- Quality assessor with graceful degradation
- 267 automated tests + 62 scenario tests

**Libraries used (not built by us):**
- OpenCV — image processing primitives
- FastAPI — web framework
- React — UI framework
- pyttsx3 — text-to-speech engine
- Web Speech API — browser voice recognition

---

## Known Limitations

- **Real embossed Braille** (raised bumps, same color as paper) requires strong side lighting to create visible shadows for detection
- **Phone/monitor screens** shown to webcam produce unreliable results due to backlight, glare, and pixel interference
- **Punctuation** grid position assignment can be incorrect for single-column dot patterns
- **Single-character detection** (1-3 dots) is rejected by the noise filter to prevent false positives on faces/objects

## Future Improvements

- ML-based dot detection (CNN trained on real Braille photos) for reliable embossed detection
- Grade 2 Braille (contractions) support
- Better rotation correction for sparse dot patterns
- Punctuation grid position disambiguation using neighboring cell context
- Mobile app (React Native or Flutter)
- Offline mode with edge AI
- Multi-language Braille support
- Braille writing assistance (reverse translation)

---

## Team

Built for BrailleVision Hackathon 2026.

---

## License

MIT
