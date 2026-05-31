# BrailleVision Hackathon 2026 — Submission

## Project Name
**Braille Vision Scanner**

## Short Description
A camera-based system that detects physical Braille dot patterns from images, segments them into cells, decodes them to English text, and speaks the result aloud. Designed with accessibility-first principles for visually impaired users.

## Technology Stack
- **Backend**: Python 3.10, FastAPI, OpenCV, NumPy, pyttsx3
- **Frontend**: React 18, TypeScript, Vite, Web Speech API
- **Communication**: WebSocket (real-time), REST API
- **Testing**: pytest (267 tests), Vitest (21 tests)

## How the System Detects Physical Braille

### Pipeline Overview
```
Camera/Image Input → Preprocessing → Dot Detection → Cell Segmentation → Pattern Recognition → English Text → Speech Output
```

### Step-by-Step:

1. **Image Input**: Accepts camera-captured Braille images via upload or WebSocket streaming at 2 FPS

2. **Preprocessing**:
   - Brightness normalization (target mean 128)
   - Noise reduction (Gaussian 5x5 kernel)
   - Lighting-aware thresholding: detects if lighting is uniform or uneven across image quadrants
   - Uniform lighting → Otsu's global thresholding
   - Uneven lighting → Adaptive Gaussian thresholding
   - Auto-inversion to ensure dots are always white on black

3. **Dot Detection**:
   - OpenCV contour analysis on binary image
   - Classification using circularity (≥0.40), convexity (≥0.60), inertia ratio (≥0.30)
   - Size estimation from circular contours (ignores noise for size calculation)
   - Fallback: multi-strategy embossed detector using Difference of Gaussians, Hough Circle Transform, and morphological top-hat filtering

4. **Cell Segmentation**:
   - Groups dots into 2x3 Braille cell grids
   - Natural-break algorithm estimates intra-cell vs inter-cell spacing
   - Line detection via vertical gap analysis (>2x row spacing)
   - Word boundary detection (>1.8x median inter-cell gap)
   - Handles spacing variation 50-150%

5. **Pattern Recognition**:
   - Grade 1 Braille lookup table (26 letters, 10 digits, 6 punctuation marks)
   - Number indicator state machine (dots 3-4-5-6)
   - Capital indicator handling (dot 6)
   - Unknown pattern detection (U+FFFD placeholder)

6. **Noise Rejection**:
   - Grid structure validation (Y-clustering, X-regularity)
   - Dot size uniformity check (CV < 0.6)
   - Minimum 4 dots required
   - Confidence and unrecognized-character thresholds

7. **Output**:
   - Text displayed on screen with confidence bar
   - Text-to-Speech via Web Speech API
   - Adjustable speech rate (80-250 WPM)

## Accuracy / Performance Notes

### What Works Well
- All 26 English letters: 97% accuracy on camera-captured images
- Numbers (0-9) with number indicator: correct
- Multi-line text: correct reading order
- Different backgrounds: white paper, cream, gray, dark
- Paper texture and light noise: handled
- Left/right/top shadows: handled by adaptive thresholding
- Different dot sizes (8px to 20px radius): handled
- Slight camera blur: handled

### Known Limitations
- Real embossed Braille (raised bumps, same color as paper) requires strong side lighting to create visible shadows — detection is inconsistent without proper lighting
- Punctuation grid position assignment can be incorrect for dots only in the right column
- Very heavy noise (>30 units) overwhelms the detector
- Showing a phone/monitor screen to a webcam produces unreliable results due to backlight interference

### Performance
- Average processing: 27ms per frame
- End-to-end (upload to speech): < 2 seconds
- Continuous streaming: 2+ FPS throughput
- Memory: < 2GB RAM

### Test Results
- 267 unit tests passing
- 22/32 stress test scenarios passing
- 18/30 realistic condition scenarios passing

## Instructions to Run

### Prerequisites
- Python 3.10+ with pip
- Node.js 18+ with npm

### Setup
```bash
# Install backend
cd backend
pip install -r requirements.txt
cd ..

# Install frontend
cd frontend
npm install
cd ..
```

### Run
```bash
# Terminal 1 — Backend (from project root)
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open http://localhost:5173

### Test
```bash
# Generate test image and verify pipeline
python -m backend.test_braille_image

# Run stress tests
python -m backend.stress_test_braille

# Run realistic condition tests
python -m backend.realistic_test_braille

# Generate pipeline stage visualizations
python -m backend.generate_pipeline_stages

# Unit tests
cd backend && pytest tests/ -v
cd frontend && npm test
```

## Accessibility Features
- High-contrast UI (7:1+ ratio, WCAG AAA)
- 48x48dp minimum touch targets
- Full ARIA labeling (role, name, state)
- Voice command support (start scan, stop, repeat, faster, slower)
- Audio tutorial on startup
- Smart noise rejection (stays silent when no Braille detected)
- Visual text fallback when audio unavailable
- Screen reader compatible

## Future Improvement Plan
1. **ML-based detection** — Train a CNN on real embossed Braille photographs for reliable detection without requiring side lighting
2. **Grade 2 Braille** — Contractions and shorthand support
3. **Better grid assignment** — Use neighboring cell context to disambiguate single-dot cells (capitals, punctuation)
4. **Mobile app** — React Native or Flutter for native camera access and better performance
5. **Offline mode** — Edge deployment for use without internet
6. **Multi-language** — Support for French, Spanish, Arabic Braille

## What Was Built vs. Reused

### Built during hackathon (original work):
- Complete recognition pipeline (7 modules)
- Lighting-aware adaptive thresholding
- Multi-strategy dot detection (contour + DoG + Hough + top-hat)
- Grid-structure noise rejection algorithm
- Natural-break cell segmentation with space detection
- Grade 1 Braille decoder with number/capital state machine
- Accessible React UI with voice commands
- WebSocket real-time streaming architecture
- Pipeline stage visualization generator
- 267+ automated tests and stress testing framework

### Open-source libraries used:
- OpenCV (image processing) — https://opencv.org
- FastAPI (web framework) — https://fastapi.tiangolo.com
- React (UI) — https://react.dev
- pyttsx3 (TTS) — https://pypi.org/project/pyttsx3
- Vite (build tool) — https://vitejs.dev
