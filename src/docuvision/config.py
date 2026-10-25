"""Pipeline configuration and named processing presets.

Syllabus: Unit 1 - image processing workflow configuration
"""

from dataclasses import asdict, dataclass
from dataclasses import replace as dataclass_replace
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable settings for a DocuVision processing run.

    Args:
        work_size: Maximum image side used for processing.
        illum: Illumination correction method.
        denoise: Denoising method.
        denoise_ksize: Kernel size used by the denoising method.
        contrast: Contrast enhancement method.
        clahe_clip: CLAHE contrast limit.
        clahe_tile: Number of tiles per side for CLAHE.
        sharpen: Sharpening method.
        sharpen_amount: Sharpening strength.
        binarize: Binarization method.
        sauvola_window: Local window size for Sauvola thresholding.
        sauvola_k: Sauvola threshold sensitivity.
        morph_cleanup: Whether to clean the binary image morphologically.
        deskew: Whether to deskew the document.
        ocr_source: Image stage supplied to OCR.

    Returns:
        A frozen collection of pipeline parameters.

    Syllabus: Unit 1 - image processing workflow configuration
    """

    work_size: int = 1600  # Bounds memory and processing cost while retaining page detail.
    illum: str = "homomorphic"  # Compensates for uneven illumination without removing detail.
    denoise: str = "median"  # Suppresses impulse noise while preserving text boundaries.
    denoise_ksize: int = 3  # The smallest useful odd median kernel limits stroke damage.
    contrast: str = "clahe"  # Local contrast helps text under nonuniform lighting.
    clahe_clip: float = 2.0  # Moderate clipping reduces noise amplification.
    clahe_tile: int = 8  # Eight tiles per side gives local adaptation without tiny regions.
    sharpen: str = "unsharp"  # Unsharp masking enhances edges with a familiar spatial method.
    sharpen_amount: float = 1.0  # Unit strength is a neutral baseline before tuning.
    binarize: str = "sauvola"  # Local thresholding handles paper illumination variation.
    sauvola_window: int = 25  # A 25-pixel window spans strokes and nearby background.
    sauvola_k: float = 0.2  # A moderate sensitivity is the requested baseline.
    morph_cleanup: bool = True  # Cleanup removes isolated noise from the binary result.
    deskew: bool = True  # Deskewing aligns text rows for downstream OCR.
    ocr_source: str = "enhanced"  # Enhanced grayscale retains useful antialiasing for OCR.
    disabled_steps: tuple[str, ...] = ()  # Named run_pipeline steps can be selectively bypassed.

    def replace(self, **kw: Any) -> "PipelineConfig":
        """Return a copy with selected fields changed.

        Args:
            **kw: Configuration fields to replace.

        Returns:
            A new immutable configuration.

        Syllabus: Unit 1 - image processing workflow configuration
        """
        return dataclass_replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        """Convert this configuration to a plain dictionary.

        Args:
            None.

        Returns:
            A dictionary containing every configuration field.

        Syllabus: Unit 1 - image processing workflow configuration
        """
        return asdict(self)

    @classmethod
    def preset(cls, name: str) -> "PipelineConfig":
        """Return a named configuration preset.

        Args:
            name: Preset identifier.

        Returns:
            The requested configuration preset.

        Syllabus: Unit 1 - image processing workflow configuration
        """
        presets = {
            "fast": cls(
                work_size=1000,
                illum="none",
                denoise="median",
                denoise_ksize=3,
                contrast="clahe",
                sharpen="none",
                binarize="otsu",
                sauvola_window=15,
                morph_cleanup=True,
                deskew=False,
            ),
            "balanced": cls(),
            "max_quality": cls(
                work_size=2400,
                illum="homomorphic",
                denoise="median",
                denoise_ksize=3,
                contrast="clahe",
                clahe_clip=2.5,
                clahe_tile=8,
                sharpen="unsharp",
                sharpen_amount=1.2,
                binarize="sauvola",
                sauvola_window=31,
                sauvola_k=0.2,
                morph_cleanup=True,
                deskew=True,
            ),
            "baseline_otsu": cls(
                illum="none",
                denoise="none",
                contrast="none",
                sharpen="none",
                binarize="otsu",
                morph_cleanup=False,
                deskew=False,
                disabled_steps=(
                    "document_rectification",
                    "illumination_correction",
                    "denoise",
                    "contrast",
                    "sharpen",
                    "morphology_cleanup",
                    "deskew",
                    "layout",
                    "ocr",
                ),
            ),
            "baseline_adaptive": cls(
                illum="none",
                denoise="none",
                contrast="none",
                sharpen="none",
                binarize="adaptive",
                morph_cleanup=False,
                deskew=False,
                disabled_steps=(
                    "document_rectification",
                    "illumination_correction",
                    "denoise",
                    "contrast",
                    "sharpen",
                    "morphology_cleanup",
                    "deskew",
                    "layout",
                    "ocr",
                ),
            ),
        }
        try:
            return presets[name]
        except KeyError as error:
            raise ValueError(
                "name must be one of 'fast', 'balanced', 'max_quality', "
                "'baseline_otsu', or 'baseline_adaptive'."
            ) from error
