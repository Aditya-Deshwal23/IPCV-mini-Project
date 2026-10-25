# DocuVision Viva Preparation

Questions and answers below follow the current implementation in
`src/docuvision/`. The evaluation sections refer to
`src/docuvision/evaluation/`.

## `config.py`

1. **Why is `PipelineConfig` frozen?** It prevents a run's settings from being
   mutated unexpectedly; `replace()` returns a new configuration.
2. **Which presets are available?** `fast`, `balanced`, `max_quality`,
   `baseline_otsu`, and `baseline_adaptive`.
3. **How is a step disabled?** Its stable step name is included in
   `disabled_steps`; the pipeline still records a skipped stage.
4. **What is `work_size` for?** It bounds the longest processed image side to
   control runtime and memory.
5. **Which binarizer does balanced use?** Sauvola, with a local window and
   sensitivity value configured in the dataclass.

## `io_utils.py`

1. **What color order does `load_image` return?** OpenCV BGR, as a three-channel
   uint8 array.
2. **What does `ensure_uint8` do?** It rejects nonnumeric or nonfinite data,
   copies uint8 input, and clips other numeric input to [0, 255].
3. **How does `resize_max_side` choose output dimensions?** It preserves the
   aspect ratio and caps the longest side.
4. **How do you request grayscale from `to_gray`?** Pass a grayscale or BGR
   image; BGR input is converted with OpenCV.
5. **How does `read_video_frames` select frames?** It yields frame zero and
   then every `stride`th decoded frame, releasing the capture when finished.

## `enhancement.py`

1. **What does the negative operation calculate?** For each byte intensity,
   \(s=255-r\).
2. **How does gamma affect dark levels?** For normalized intensity \(r\),
   \(s=r^\gamma\): gamma below one expands dark values; gamma above one
   compresses them.
3. **Why use CLAHE instead of global equalization?** CLAHE equalizes local
   tiles and clips histogram amplification to limit noise growth.
4. **Why use a median filter for impulse noise?** It rejects isolated extremes
   while preserving many edge transitions better than averaging.
5. **How is unsharp masking formed?** It adds a scaled difference between the
   original and a Gaussian-smoothed image.

## `geometry.py`

1. **How are detected document corners ordered?** Top-left, top-right,
   bottom-right, bottom-left.
2. **What does `affine_transform` return?** The transformed image and a 3×3
   homogeneous transform matrix.
3. **How does affine mapping differ from perspective rectification?** Affine
   mapping uses a 2D affine matrix; perspective rectification uses a
   projective homography from four corner correspondences.
4. **What happens when no document quadrilateral is found?** Corner detection
   raises `ValueError`; the pipeline catches it and uses its documented
   fallback.
5. **Does `affine_transform` mutate its input?** No; OpenCV writes an output
   array and the input is preserved.

## `segmentation.py`

1. **When do 4- and 8-connectivity produce different components?** Diagonal
   foreground pixels are separate under 4-connectivity but connected under
   8-connectivity.
2. **What does Otsu maximize?** Between-class variance over candidate global
   thresholds.
3. **Why can Sauvola beat global Otsu on shadows?** Its threshold is computed
   from local mean and standard deviation, so it can follow slowly varying
   background brightness.
4. **How does region growing decide whether to add a neighbor?** It compares
   the neighbor's intensity with the seed intensity using the given tolerance
   and connectivity.
5. **How does watershed split touching objects?** It finds foreground
   distance peaks for markers and floods the distance landscape to assign
   labels.

## `frequency.py`

1. **What is the sign in the forward DFT exponential?** The matrix uses
   \(e^{-2\pi i kn/N}\); the inverse uses the positive sign and divides by N.
2. **Why use `fftshift` for display and masks?** It moves the DC/zero-frequency
   bin to the center, making radial filters and spectrum viewing intuitive.
3. **Why can an ideal lowpass cause ringing?** Its abrupt frequency cutoff
   corresponds to a long oscillatory spatial response; smooth Butterworth and
   Gaussian roll-offs soften that effect.
4. **Why are highpass masks explicit complements?** Each implementation
   returns `1 - lowpass`, so same-family lowpass and highpass sum to one.
5. **What image model motivates homomorphic filtering?** Image intensity is
   modeled as illumination times reflectance; taking a logarithm converts
   that product into a sum of low- and higher-frequency contributions.

## `morphology.py`

1. **What is a structuring element?** A chosen neighborhood footprint used to
   decide how each pixel is updated by a morphological operation.
2. **How do erosion and dilation differ?** Erosion shrinks bright foreground;
   dilation expands it.
3. **What does opening do?** Erosion followed by dilation, commonly removing
   small bright foreground specks.
4. **What does closing do?** Dilation followed by erosion, commonly filling
   small gaps in bright foreground.
5. **How are the scratch implementations checked?** The NumPy sliding-window
   outputs are compared against the corresponding OpenCV morphology results
   in tests.

## `features.py`

1. **What is the Harris response?** \(R=\det(M)-k\,\mathrm{trace}(M)^2\),
   where \(M\) is the smoothed local gradient covariance matrix.
2. **What does Shi-Tomasi score?** The smaller eigenvalue of the local
   gradient covariance matrix; OpenCV's `goodFeaturesToTrack` selects these
   corners.
3. **Why is DoG useful for scale-space features?** Adjacent Gaussian levels
   produce a Difference-of-Gaussian response that approximates LoG.
4. **What is Lowe's ratio test checking?** It rejects an ambiguous match if
   the best descriptor distance is not sufficiently smaller than the second
   best.
5. **Why use RANSAC for a homography?** It estimates the transform while
   rejecting inconsistent feature matches as outliers.

## `motion.py`

1. **What is the aperture problem?** A straight edge constrains motion only
   perpendicular to the edge, leaving motion along it ambiguous; corners
   provide gradients in multiple directions.
2. **How does pyramidal Lucas-Kanade help?** It estimates large movement at
   coarse resolution and refines at finer image scales.
3. **What validates OpenCV LK tracks here?** A forward-backward track must
   return within one pixel, in addition to OpenCV's status flags.
4. **What extra assumption distinguishes Horn-Schunck from local LK?**
   Horn-Schunck regularizes the flow field to be spatially smooth.
5. **Which frame is emitted after a stable pause?** The frame with the
   greatest variance of the Laplacian among the stable candidates.

## `compression.py`

1. **What does the RLE stream store?** A header with dimensions and first
   pixel, followed by encoded run lengths (or value/count pairs for gray RLE).
2. **Why are Huffman codes canonical?** The decoder can reconstruct the same
   prefix code from the stored code lengths and symbol ordering.
3. **Why is Huffman optimal?** For the known symbol frequencies, its tree
   minimizes expected codeword length among binary prefix codes.
4. **How does the LZW dictionary reset?** When the 12-bit code space is full,
   the encoder writes a clear marker and both sides return to the initial
   byte dictionary.
5. **How does the UI calculate compression ratio?** It divides input bytes by
   the actual serialized stream length, including headers.

## `ocr.py`

1. **How can the executable path be configured?** Set `TESSERACT_CMD`; the
   module configures pytesseract to use it.
2. **Does missing Tesseract crash a scan?** No; OCR returns a clear
   unavailable status and an empty word table.
3. **What is in the word table?** Tesseract word text, bounding-box
   coordinates and dimensions, and confidence values.
4. **How are CER and WER normalized?** Text is lowercased and whitespace is
   collapsed before Levenshtein distance is divided by reference length.
5. **What does `psm=6` request?** Tesseract's page segmentation mode for a
   uniform block of text.

## `pipeline.py`

1. **Why are intermediate stages retained?** The UI can display every
   technique, associated unit, parameters, elapsed time, and skipped state.
2. **What is the document-geometry fallback order?** Detected corners and
   four-point warp, then SIFT template alignment when supplied, then the
   original full image.
3. **How are disabled steps represented?** They return a copy of the current
   image and add a stage with `skipped=True` and zero seconds.
4. **How does the pipeline produce a binary result?** The configured
   binarizer creates a 0/255 image, followed optionally by morphology and
   deskew.
5. **How is OCR input selected?** `ocr_source` determines whether the
   enhanced grayscale or binary stage is sent to `run_ocr`.

## `evaluation/degrade.py`

1. **Why does each degradation accept a seed?** To make random noise and
   geometric perturbations reproducible.
2. **Which shadows are supported?** Smooth linear and radial illumination
   fields.
3. **What does SET_A preserve?** The clean/degraded pixel alignment, so
   full-frame PSNR and SSIM remain meaningful.
4. **What ground truth accompanies perspective degradation?** The generated
   true page-corner coordinates.
5. **How are JPEG artifacts introduced?** The image is encoded and decoded
   at a selected JPEG quality.

## `evaluation/metrics.py`

1. **What does infinite PSNR mean?** Mean squared error is zero because the
   images match exactly.
2. **What is the data range supplied to SSIM?** 255 for the project's uint8
   image representation.
3. **How is corner error calculated?** Mean Euclidean distance between four
   corresponding predicted and true corners.
4. **What does the sharpness metric measure?** Variance of the grayscale
   Laplacian response.
5. **What does RMS contrast measure?** The standard deviation of grayscale
   pixel intensities.

## `evaluation/make_dataset.py`

1. **What is a generated clean page's size?** 1240×1754 pixels.
2. **Where does page text come from?** A built-in list of ten English
   passages, selected with a seeded random generator.
3. **How does font loading remain portable?** It tries common system
   TrueType paths and falls back to Pillow's default font.
4. **How do SET_A and SET_B differ?** SET_A has photometric changes only;
   SET_B also applies perspective and rotation over a generated background.
5. **What ground truth is saved?** Text files are saved beside the page
   images; SET_B also stores true corner coordinates in JSON.

## `evaluation/run_eval.py`

1. **Which SET_A methods are compared?** Raw, Otsu-only, adaptive-only, and
   full pipeline.
2. **When are CER/WER columns omitted?** When Tesseract is unavailable.
3. **What is an ablation run?** A pipeline run with one selected stage
   disabled, compared with the full setup.
4. **How does the compression study avoid comparing only theoretical rates?**
   It records byte lengths from actual encoded streams and reports
   round-trip losslessness.
5. **Which artifacts are produced?** Per-sample and aggregate CSVs, PNG
   charts, and a generated `summary.md`.

## `app.py`

1. **Where does the shared image come from?** The sidebar decodes an upload,
   camera image, or selected sample and stores it in `st.session_state`.
2. **How many lab tabs are wired?** Eight: Scan, Units 1–6, and Evaluation.
3. **Why is there a Tesseract banner?** OCR depends on an external executable;
   the app warns users before they reach an OCR-dependent workflow.
4. **Where are the algorithms implemented?** In `src/docuvision/`; the app
   dispatches to the tab renderers.
5. **How is the pipeline preset shared?** The selected sidebar preset is
   converted to a `PipelineConfig` and passed to the Scan tab.

## `ui/common.py`

1. **What does `require_image` accept?** Nonempty uint8 grayscale,
   singleton-channel, or BGR arrays; it shows a message for missing input.
2. **Why does the helper `gray` check dimensionality first?** Grayscale input
   can be returned directly instead of mistakenly passed through a BGR
   conversion.
3. **What does `rgb` change?** It converts grayscale to three-channel RGB and
   BGR to RGB for Streamlit's display convention.
4. **What information starts each tab?** `syllabus_info` displays the
   syllabus topics and experiment label.
5. **Does `show_pair` mutate its images?** No; it only creates display
   conversions and arranges two columns.

## `ui/tab_scan.py`

1. **What keys the pipeline cache?** Source bytes, array shape, and stable
   serialized pipeline configuration.
2. **Why are stage images shown?** They make each processing step and its
   timing/parameters inspectable.
3. **How does the tab calculate optional CER?** It compares the entered
   ground truth with the pipeline OCR text using the project's metric helper.
4. **Which files can the user download?** Final PNG, recognized TXT, and
   single-page PDF.
5. **How is an unavailable OCR result communicated?** The pipeline status and
   explicit returned text are shown instead of raising an app exception.

## `ui/tab_enhancement.py`

1. **Which operations can be selected?** Point transforms, histogram
   operations, bit planes, spatial filters, sharpening, and image arithmetic.
2. **Why is the filter kernel bounded?** Small images require a kernel no
   larger than their extent; a one-pixel image uses a safe identity-sized
   kernel.
3. **How is the noise comparison made reproducible?** The noise control passes
   a user-visible seed to the enhancement library.
4. **What measures filter quality in the comparison table?** PSNR against the
   original clean grayscale image.
5. **What do the sampling controls demonstrate?** Spatial downsampling and
   reducing the number of available intensity levels.

## `ui/tab_segmentation.py`

1. **Which gradient operators are available?** Sobel, Prewitt, and Roberts,
   returning horizontal/vertical response, magnitude, and direction.
2. **How are line-mask orientations selected?** The UI chooses among the
   horizontal, vertical, 45-degree, and 135-degree textbook masks.
3. **What local threshold methods are exposed?** Adaptive Gaussian and
   Sauvola, in addition to global Otsu.
4. **What is plotted for connectivity?** Separate 4- and 8-connected label
   maps and their foreground component counts.
5. **Why may split-and-merge be unavailable?** Its library algorithm is
   limited to images no larger than 256 pixels per side.

## `ui/tab_frequency.py`

1. **Which frequency filters can be selected?** Ideal, Butterworth, or
   Gaussian, each as lowpass or highpass.
2. **What is shown in the spectrum section?** The source, centered log
   magnitude, and phase display.
3. **How is the ringing image generated?** The library filters a synthetic
   step edge with ideal, Butterworth, and Gaussian lowpass masks.
4. **How does the notch example create periodic noise?** It adds a horizontal
   cosine pattern, then applies a centered conjugate-paired reject mask.
5. **How does the WHT reconstruction control work?** It retains the selected
   largest-magnitude coefficients and applies the inverse orthonormal WHT.

## `ui/tab_features.py`

1. **Why are images reduced before feature visualization?** The UI caps the
   longest side at 256 pixels to keep scratch and feature operations
   responsive.
2. **What matrix is displayed for geometry?** The 3×3 homogeneous matrix
   returned by the library's affine transform.
3. **What happens if document corners are not found?** The UI displays a
   no-corners message and continues to feature demonstrations.
4. **How are template features matched?** SIFT descriptors use a
   nearest-neighbor ratio test; template alignment is estimated by the
   library's RANSAC homography path.
5. **What scale-space results can be inspected?** Gaussian pyramid levels,
   reconstructed Laplacian output, and selected DoG octave/scale images.

## `ui/tab_motion.py`

1. **How is progress obtained from `process_video`?** The tab advances its
   generator frame-by-frame and updates a Streamlit progress bar from yielded
   progress values.
2. **Why can it use a generated video?** `make_demo_video` creates a local
   deterministic motion sequence without external assets.
3. **How are captures shared with Scan?** A selected captured frame is copied
   into the shared source image session state.
4. **What dense and sparse flow displays are provided?** LK arrows for
   Shi-Tomasi points and Horn-Schunck color/arrows for dense flow.
5. **What happens on a tiny still image?** Adaptive-threshold or watershed
   controls show a size message where their neighborhood needs more pixels.

## `ui/tab_compression.py`

1. **Are codec byte counts estimated?** No; the comparison encodes and
   decodes actual streams, then reports serialized lengths including headers.
2. **How are large inputs made responsive?** The UI area-resizes the grayscale
   scan to at most 256 pixels per side.
3. **What does the binary-input checkbox change?** It selects the Otsu
   0/255 image as the primary source for entropy and transform displays.
4. **How is the Huffman code table generated?** It uses code lengths and
   canonical codes from the same byte-frequency model as the encoder.
5. **What does the rate-distortion chart compare?** Measured compression
   ratio and PSNR for WHT and DCT at several quantizer steps.

## `ui/tab_evaluation.py`

1. **How does the tab find evaluation artifacts?** It recursively scans
   `results/` for CSV, PNG, and summary Markdown files.
2. **What does the rerun button execute?** A deterministic N=2 evaluation
   into separate `data/evaluation-smoke/` and `results/small-run/` folders.
3. **How are CSV values displayed?** As data tables and charts for numeric
   columns.
4. **What happens if a CSV is malformed?** The tab displays an explicit
   error and continues to the other artifacts.
5. **Where does the generated summary originate?** From the evaluator's
   CSV-backed `summary.md`, not a table manually typed into the UI.

## `scripts/sync_results_docs.py`

1. **What does the script synchronize?** `results/summary.md` into marked
   blocks in the README and technical report.
2. **How does it avoid replacing unrelated document content?** It replaces
   only the text between the unique start and end markers.
3. **What happens when the summary is missing?** It raises a clear
   `FileNotFoundError` telling the user to run evaluation first.
4. **How are accidental duplicate markers handled?** The script raises
   `ValueError` instead of guessing which block to replace.
5. **Is repeated synchronization stable?** Yes; the test verifies it is
   idempotent when the source summary has not changed.

## `live_demo.py`

1. **Which input is used by default?** Camera index zero; `--video path` selects
   a replay file instead.
2. **How does the demo get a document quadrilateral?** `DocumentTracker`
   detects corners on the initial frame and tracks in-page Shi-Tomasi points.
3. **What triggers automatic capture?** The stability detector observes its
   configured number of consecutive low-motion frames and returns its sharpest
   stable candidate.
4. **What happens to a captured frame?** It is passed through `run_pipeline`;
   rectified and enhanced images are saved under `results/live/`.
5. **What do the keyboard keys do?** **q** quits; **s** manually sends the
   current frame through the same processing and save path.
