# Demo Video Script (2-3 minutes)

## Opening (15 seconds)
"This is Braille Vision Scanner — a system that detects physical Braille dot patterns from camera-captured images and converts them to spoken English."

**Show**: App landing page with high-contrast UI

---

## Part 1: Upload Demo (45 seconds)

**Action**: Click "Upload Image" → select `braille_test_sheet.png`

**Narrate**: "I'm uploading a camera-captured image of physical Braille dots. The system processes it through a computer vision pipeline — preprocessing, dot detection, cell segmentation, and pattern decoding."

**Show**: The decoded text appears: "hello, abc, no, ok". TTS speaks the result.

**Action**: Upload `test_braille_hello.png`

**Narrate**: "It also handles different backgrounds — here's Braille on a dark surface. The adaptive thresholding automatically adjusts."

**Show**: Decoded "hello" with 97% confidence.

---

## Part 2: Pipeline Visualization (30 seconds)

**Action**: Show the `pipeline_stages/` images in sequence

**Narrate**: "Let me walk through what happens at each stage:
1. Original image captured by camera
2. Convert to grayscale
3. Normalize brightness
4. Reduce noise
5. Binary thresholding separates dots from background
6. Contour analysis detects individual dots — shown here in green
7. Dots are grouped into 2x3 Braille cells
8. Each cell pattern maps to an English character"

---

## Part 3: Live Camera Features (30 seconds)

**Action**: Click "Start Scan", show camera active

**Narrate**: "The system also supports live camera streaming via WebSocket at 2 frames per second. It includes smart noise rejection — watch what happens when I point at random objects."

**Show**: Camera pointed at face/desk → status shows "Scanning... (no Braille detected)" — silence.

**Narrate**: "It stays silent because the grid-structure validation rejects anything that isn't a regular dot pattern. A blind user won't be bombarded with false readings."

---

## Part 4: Accessibility & Voice Commands (30 seconds)

**Show**: High-contrast UI, large buttons, voice commands list

**Narrate**: "The interface is built for visually impaired users — 7:1 contrast ratio, 48-pixel touch targets, full ARIA labeling, and voice commands."

**Action**: Say "repeat" → hears last result again

**Narrate**: "Users can control everything hands-free: start scan, stop scan, repeat, faster, slower."

---

## Part 5: Technical Highlights (20 seconds)

**Narrate**: "Key technical features:
- Adaptive thresholding that switches between Otsu and Gaussian based on lighting uniformity
- Multi-strategy dot detection including Difference of Gaussians and Hough Circle Transform for embossed Braille
- Grid-structure validation that rejects noise with less than 5% false positive rate
- 27 milliseconds average processing per frame
- 267 automated tests covering all pipeline stages"

---

## Closing (10 seconds)

**Narrate**: "Braille Vision Scanner — making physical Braille accessible through computer vision. Built with Python, OpenCV, FastAPI, and React."

---

## Tips for Recording

1. Use OBS Studio or Windows Game Bar (Win+G) to record
2. Record at 1080p minimum
3. Keep it under 3 minutes
4. Speak clearly at moderate pace
5. Show the terminal briefly to prove it's running locally
6. Test everything before recording — do a dry run
