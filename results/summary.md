# Reproducible DocuVision Evaluation

- Pages per set: 2
- Dataset seed: 2026
- Tesseract available: False
- Image metrics compare each photometric SET_A output with its pixel-aligned clean page.
- Compression ratios use actual encoded byte streams, including codec headers.

## SET_A quality and OCR metrics (mean and standard deviation)

| method | metric | mean | std | n |
| --- | --- | --- | --- | --- |
| raw | psnr | 11.8149 | 0.0402 | 2 |
| raw | ssim | 0.8944 | 0.0014 | 2 |
| otsu_only | psnr | 15.4232 | 16.1489 | 2 |
| otsu_only | ssim | 0.7454 | 0.3209 | 2 |
| adaptive_only | psnr | 8.7371 | 0.1048 | 2 |
| adaptive_only | ssim | 0.1125 | 0.0261 | 2 |
| full_pipeline | psnr | 4.6683 | 0.2997 | 2 |
| full_pipeline | ssim | 0.6068 | 0.0159 | 2 |

![SET_A quality](set_a_quality.png)

## SET_B geometry and OCR

| metric | mean | std | n |
| --- | --- | --- | --- |
| corner_error_px | — | 0.0000 | 0 |
| sharpness | 103.5791 | 36.9117 | 2 |

## SET_A stage ablation

| omitted_stage | metric | mean | std | n |
| --- | --- | --- | --- | --- |
| illumination_correction | psnr | 11.6127 | 0.2227 | 2 |
| illumination_correction | ssim | 0.8691 | 0.0086 | 2 |
| denoise | psnr | 4.7388 | 0.2994 | 2 |
| denoise | ssim | 0.5505 | 0.0202 | 2 |
| contrast | psnr | 4.8098 | 0.3369 | 2 |
| contrast | ssim | 0.6589 | 0.0180 | 2 |
| sharpen | psnr | 4.6603 | 0.2964 | 2 |
| sharpen | ssim | 0.6212 | 0.0145 | 2 |
| binarize | psnr | 4.6683 | 0.2997 | 2 |
| binarize | ssim | 0.6068 | 0.0159 | 2 |
| morphology_cleanup | psnr | 4.6683 | 0.2997 | 2 |
| morphology_cleanup | ssim | 0.6068 | 0.0159 | 2 |

## Actual-byte codec comparison

| codec | ratio_mean | ratio_std | bytes_mean | lossless_fraction |
| --- | --- | --- | --- | --- |
| binary_jpeg_10 | 83.9558 | 0.0000 | 25906.0000 | 1.0000 |
| binary_jpeg_50 | 83.9526 | 0.0000 | 25907.0000 | 1.0000 |
| binary_jpeg_90 | 83.9526 | 0.0000 | 25907.0000 | 1.0000 |
| binary_png | 424.7139 | 0.0000 | 5121.0000 | 1.0000 |
| gray_jpeg_10 | 56.7417 | 3.9076 | 38422.0000 | 0.0000 |
| gray_jpeg_50 | 25.6869 | 2.2188 | 84989.0000 | 0.0000 |
| gray_jpeg_90 | 8.0236 | 0.4024 | 271413.0000 | 0.0000 |
| gray_png | 3.0061 | 0.0676 | 723710.0000 | 1.0000 |
| huffman_binary | 86998.4000 | 0.0000 | 25.0000 | 1.0000 |
| huffman_gray | 1.2504 | 0.0574 | 1741300.0000 | 1.0000 |
| lzw_binary | 691.7812 | 0.0000 | 3144.0000 | 1.0000 |
| lzw_gray | 2.1153 | 0.0686 | 1028763.5000 | 1.0000 |
| rle_binary | 127938.8235 | 0.0000 | 17.0000 | 1.0000 |
| rle_bitplanes | 1.0294 | 0.0335 | 2114054.0000 | 1.0000 |
| rle_gray | 1.1808 | 0.0191 | 1842215.0000 | 1.0000 |

## WHT vs DCT rate-distortion

| transform | parameter | CR_mean | PSNR_mean | PSNR_std | SSIM_mean |
| --- | --- | --- | --- | --- | --- |
| dct | keep=16 | 2.3143 | 32.4387 | 4.0369 | 0.8969 |
| dct | keep=64 | 1.4574 | 59.0349 | 0.0023 | 0.9992 |
| dct | qstep=24.0 | 3.2753 | 37.2601 | 0.7386 | 0.9605 |
| dct | qstep=8.0 | 2.6504 | 44.8948 | 0.3303 | 0.9883 |
| wht | keep=16 | 2.2741 | 30.1205 | 1.8822 | 0.8994 |
| wht | keep=64 | 1.4080 | 58.1975 | 0.0514 | 0.9991 |
| wht | qstep=24.0 | 3.1923 | 36.8219 | 0.8014 | 0.9530 |
| wht | qstep=8.0 | 2.5662 | 44.7919 | 0.2270 | 0.9884 |

![Rate-distortion](rate_distortion.png)

Raw per-page data is available in the adjacent CSV files.
