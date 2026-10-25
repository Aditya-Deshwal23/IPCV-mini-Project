# DocuVision coding instructions

- Use classical image processing only; do not add deep-learning models or pretrained models.
- Keep library code in `src/docuvision/` pure: arrays and parameters in, arrays or dataclasses out. Do not use Streamlit, print, file dialogs, or global state there. The only stateful classes are `DocumentTracker` and `StabilityDetector` in `motion.py`.
- Treat images as `uint8` BGR unless a function documents another format. Never mutate input arrays.
- Give every function full type hints and a docstring that covers purpose, arguments, returns, and a `Syllabus: Unit X - topic` line.
- Explain parameter choices in comments where parameters are introduced.
- For key syllabus topics, provide both a NumPy implementation whose function name ends in `_scratch` and an OpenCV or scikit-image implementation, with tests comparing their outputs.
- Keep scratch implementations small-image friendly; UI callers should resize to at most 256 pixels per side.
- Add pytest tests for modules using small synthetic arrays. Tests must not require downloads or internet access.
- Return intermediate pipeline images so the UI can display each stage.
- Derive README and report metrics from CSV files produced by `run_eval`; do not invent results.
- Cite only the approved sources: Otsu (1979), Lucas and Kanade (1981), Horn and Schunck (1981), Lowe (2004), Sauvola and Pietikäinen (2000), and Gonzalez and Woods.
