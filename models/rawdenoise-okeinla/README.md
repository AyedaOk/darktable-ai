# Okeinla 1.0

Okeinla jointly denoises and demosaics Bayer RAW images. It uses NAFNet with
a packed-Bayer input and a pixel-shuffle camera-RGB output.

## Source

- [Training code](https://github.com/AyedaOk/okeinla/tree/a9efc414d65fe6bd4be1c4e882be757263740d0c).
- [Checkpoint](https://huggingface.co/AyedaOk/okeinla/tree/7d4480d305ac9a425832afb9c110ee0b1af0200a).
- [Training dataset](https://huggingface.co/datasets/AyedaOk/okeinla-dataset).
- Credits: [RawNIND](https://github.com/trougnouf/rawnind_jddc),
  [NAFNet](https://github.com/megvii-research/NAFNet), and
  [RawForge](https://github.com/rymuelle/RawForge) for the training losses,
  with the [Restorer loss recipe](https://github.com/rymuelle/Restorer).

## ONNX model

`model_bayer.onnx` takes `input`, float32 `[1, 4, 512, 512]`, and returns
`output`, float32 `[1, 3, 1024, 1024]` camera RGB.
Normalize black/white levels and pack RGGB channels without input white balance.
Use cropped mirror padding, tiled inference, and output gain matching. Apply
camera color conversion after inference. The graph takes no ISO input.

## Selection criteria

| Property | Value |
| --- | --- |
| Model license | GPL-3.0-only; NAFNet components retain their MIT notice |
| OSAID v1.0 | Open Weights; no claim of full OSAID compliance |
| MOF | Class II — Open Tooling Model; self-assessed using the [Model Openness Tool](https://mot.isitopen.ai/model/evaluate) on 2026-11-20. [Assessment YAML](https://github.com/AyedaOk/okeinla/blob/8ce1e395ea1fc60d0682466ba453a349eb7cdcab/docs/okeinla-1.0-29m.yml) |
| Training data license | CC BY-SA 4.0 for both datasets; [RawNIND license record](https://doi.org/10.14428/DVN/DEQCIM) |
| Training data provenance | RawNIND and Okeinla dataset: paired noisy and reference RAW photographs. Okeinla captures use a tripod-mounted Sony A7C II with base-ISO references. The Okeinla training images show only objects. No people appear in them. |
| Training code | Public GPLv3 source linked above |
| Known limitations | Bayer only; no X-Trans or linear-RGB variant; no bitwise retraining claim. We used a frozen, ImageNet-pretrained VGG16 to compute the perceptual loss during training. We cannot verify the provenance of every image used to pretrain VGG16. Okeinla does not ship or run VGG16 at inference. |
| Published research | [Okeinla technical report](#technical-report); upstream [RawNIND](https://arxiv.org/abs/2501.08924) and [NAFNet](https://arxiv.org/abs/2204.04676) papers |
| Inference | Local; no telemetry or cloud inference |
| Scope | RAW denoising and demosaicing |
| Reproducibility | Published ONNX downloaded from an immutable Hugging Face revision |

## Build

```sh
uv run dtai run rawdenoise-okeinla
```

The demo reuses RawNIND's Bayer pipeline and skips non-Bayer samples.
The package includes GPLv3 and NAFNet MIT license texts.

## Technical report

### Architecture and training

Okeinla 1.0 uses a width-32 NAFNet with 29,162,604 parameters. The encoder has
four stages with 2, 2, 4, and 8 blocks; the middle has 12 blocks; each decoder
stage has 2 blocks. A pixel-shuffle head produces camera RGB at sensor resolution.
See the pinned [architecture source](https://github.com/AyedaOk/okeinla/blob/a9efc414d65fe6bd4be1c4e882be757263740d0c/src/rawnind/models/nafnet.py).

Training uses paired RAW photographs from RawNIND and the Okeinla dataset.
The Okeinla dataset contains 106 Sony A7C II scenes, 761 noisy exposures, and
368 base-ISO references. The configurations define scene-level validation and
test reserves, 256-pixel training crops, four crops per image, batch size four,
and output gain matching. They use seed 2026 with deterministic mode disabled.

Training losses are adapted from [RawForge](https://github.com/rymuelle/RawForge),
with the [Restorer loss recipe](https://github.com/rymuelle/Restorer).
The loss combines shadow-weighted L1 (0.16), MS-SSIM loss (0.84), and frozen
ImageNet VGG16 feature L1 loss. The final stages increase the feature-loss
weight from 0.025 to 0.035. VGG16 serves only as a training loss network;
inference needs no VGG16 weights. See the pinned
[loss implementation](https://github.com/AyedaOk/okeinla/blob/a9efc414d65fe6bd4be1c4e882be757263740d0c/src/common/libs/pt_losses.py).

The released lineage runs from M5B checkpoint 2915000 to M8 checkpoint 2925000,
then continuation checkpoint 2940000. M8 resets Adam at learning rate `1e-5`;
the continuation retains its optimizer state. The published
[configurations](https://github.com/AyedaOk/okeinla/tree/a9efc414d65fe6bd4be1c4e882be757263740d0c/src/rawnind/config/) record these stages.
Checkpoint labels do not establish the number of training updates, and the
release does not claim an exact replay of historical training.

### Evaluation and limits

We compared Okeinla and RawNIND on 133 noisy Sony A7C II images from the
[benchmark dataset](https://huggingface.co/datasets/AyedaOk/rawbench-dataset),
using one base-ISO reference per scene. All 133 test images come from the same
camera as Okeinla's own training data, while RawNIND never saw that camera.

| Model | PSNR ↑ | MS-SSIM ↑ | HFEN ↓ | Mean ΔE00 ↓ |
| --- | ---: | ---: | ---: | ---: |
| RawNIND | 47.302874 | 0.996527 | 0.579897 | 1.383124 |
| Okeinla | **47.544281** | **0.996969** | **0.565079** | **1.116526** |

The [Python script](https://github.com/AyedaOk/okeinla/blob/e4eeeaeaef1f93ae9ca96e3b30973544bec53cfe/benchmark/benchmark.py) evaluates ONNX RGB predictions
with source-derived per-tile gain correction and overlap blending. Both models
use the same color conversion, alignment, and exposure correction. We demosaic
the base-ISO RAW reference with bilinear interpolation to compare it with the
models' RGB output.

See the [benchmark instructions](https://github.com/AyedaOk/okeinla/blob/e4eeeaeaef1f93ae9ca96e3b30973544bec53cfe/benchmark/benchmark.md) for setup, pinned
dataset revision, reference selections, and report generation. The generated
`report.md` records model hashes, package versions, and metric settings.
