# Five-Minute DocuVision Demonstration

## Before the demo

- Install the project from `requirements.txt`.
- Start the app with `streamlit run app.py`.
- For the live capture portion, connect a camera and place a printed page on a
  contrasting surface. If no camera is available, use
  `python live_demo.py --video path/to/video.mp4`.
- Tesseract is optional; the app displays an availability notice if absent.

## 0:00–1:00 — Live webcam capture

1. Run `python live_demo.py`.
2. Show the green tracked page quadrilateral, number of tracked points, capture
   state, motion, and sharpness.
3. Move the page slightly, then hold it steady to demonstrate the transition
   from MOVING to STABLE and automatic capture of the sharpest stable frame.
4. Press **s** to demonstrate manual capture. Mention that warped and enhanced
   images are saved under `results/live/`.
5. Press **q** to close the live window and switch to the Streamlit app.

## 1:00–2:00 — Scan stage slider

1. In the **Scan** tab, choose or upload the same page and select the balanced
   preset.
2. Run the pipeline and compare source and processed output.
3. Move the stage slider to show the stage image, unit, technique, parameters,
   and elapsed time.
4. Show OCR status, optional ground-truth CER entry, and PNG/TXT/PDF downloads.

## 2:00–3:00 — Frequency lab and ringing

1. Open **Unit 3 · Frequency & Morphology** and show the centered spectrum.
2. Select the ideal lowpass and use the ringing demonstration to show the
   overshoot near a step edge.
3. Change to Butterworth or Gaussian lowpass and compare the smoother
   transition; briefly show a highpass complement.
4. Mention that the API pads before FFT filtering to reduce wrap-around.

## 3:00–4:00 — Motion timeline

1. Open **Unit 5 · Motion**, click **Generate demo video**, then analyze it.
2. Show motion and sharpness over the frame timeline and point out CAPTURE
   markers and captured frames.
3. Demonstrate sending a captured frame to the shared Scan source.
4. Show the LK feature arrows and Horn-Schunck color visualization if captures
   are available.

## 4:00–4:35 — Compression table

1. Open **Unit 6 · Compression** and run the byte-level codec comparison.
2. Compare actual serialized byte counts, ratios, bits per pixel, and the
   measured lossless flag.
3. Show the Huffman code table and briefly vary the WHT/DCT coefficient or
   quantization controls.

## 4:35–5:00 — Evaluation charts

1. Open **Evaluation** and show existing generated CSV-backed charts and
   summary, or run the N=2 smoke evaluation if no results are present.
2. Explain that the report uses generated CSV metrics; OCR scores are omitted
   when Tesseract is not installed.
3. Close by noting the synthetic evaluation is reproducible but not a
   substitute for a larger real-world capture study.
