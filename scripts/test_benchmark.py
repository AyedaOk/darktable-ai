"""Synthetic checks only: no network, RAW photos, or ONNX models."""

import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import benchmark
from benchmark_alignment import apply_rgb_alignment, estimate_rgb_alignment


class FakeSession:
    def get_inputs(self):
        return [type("Input", (), {"name": "input"})()]

    def run(self, names, inputs):
        gray = inputs["input"][0, 0]
        rgb = np.repeat(np.repeat(gray, 2, axis=0), 2, axis=1)
        scale = -0.2 if gray.mean() < 0.3 else 5.0
        return [np.stack([rgb] * 3)[None] * scale]


class BenchmarkTests(unittest.TestCase):
    def test_tiling_recovers_signed_per_tile_scale_and_covers_edges(self):
        gray = np.linspace(0.05, 0.6, 1200, dtype=np.float32)[None, :].repeat(550, axis=0)
        source = np.stack([gray] * 4)
        output = benchmark.infer(source, FakeSession())
        expected = np.repeat(np.repeat(gray, 2, axis=0), 2, axis=1)
        np.testing.assert_allclose(output, np.stack([expected] * 3), rtol=2e-6, atol=1e-7)

    def test_small_input_excludes_reflected_padding_from_gain(self):
        gray = np.full((32, 48), 0.2, dtype=np.float32)
        output = benchmark.infer(np.stack([gray] * 4), FakeSession())
        self.assertEqual(output.shape, (3, 64, 96))
        np.testing.assert_allclose(output, 0.2, rtol=1e-6)

    def test_zero_and_nonfinite_model_outputs_fail(self):
        source = np.full((4, 16, 16), 0.2, dtype=np.float32)
        for value in (0, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                benchmark.recover_gain(source, np.full((3, 32, 32), value, dtype=np.float32))

    def test_alignment_uses_one_source_shift_and_exposure(self):
        rng = np.random.default_rng(2026)
        gray = cv2.GaussianBlur(rng.uniform(0.05, 0.6, (512, 512)).astype(np.float32),
                               (0, 0), 2)
        gt = np.stack([gray] * 3)
        source = np.roll(gt, (4, -3), axis=(1, 2)) * 0.8
        alignment = estimate_rgb_alignment(gt, source)
        aligned_gt, aligned = apply_rgb_alignment(gt, source, alignment)
        self.assertEqual((alignment.dy, alignment.dx), (-4, 3))
        self.assertAlmostEqual(alignment.gain, 1.25, places=5)
        np.testing.assert_allclose(aligned_gt, aligned, atol=1e-7)

    def test_color_conversion_preserves_reference_neutral(self):
        profile = (np.ones(4), np.linalg.inv(benchmark.XYZ_TO_SRGB))
        rgb = np.full((3, 16, 16), 0.25, dtype=np.float32)
        np.testing.assert_allclose(benchmark.convert_color(rgb, profile), rgb, atol=1e-6)

    def test_identical_metrics(self):
        rng = np.random.default_rng(2026)
        image = rng.uniform(0.05, 0.6, (3, 192, 192)).astype(np.float32)
        values = benchmark.metrics(image, image.copy())
        self.assertEqual(values["PSNR"], float("inf"))
        self.assertAlmostEqual(values["MS-SSIM"], 1, places=6)
        self.assertEqual(values["HFEN"], 0)
        self.assertAlmostEqual(values["Mean ΔE00"], 0, places=10)

    def test_download_rejects_bad_cache_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ARW"
            path.write_bytes(b"original")
            with patch("urllib.request.urlopen") as network:
                with self.assertRaises(RuntimeError):
                    benchmark.download("https://unused.invalid", path, "0" * 64)
                network.assert_not_called()
            self.assertEqual(path.read_bytes(), b"original")

    def test_download_checks_hash_before_publishing(self):
        import io

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ARW"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"broken")):
                with self.assertRaises(RuntimeError):
                    benchmark.download("https://unused.invalid", path, "0" * 64)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".ARW.partial").exists())
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"valid")):
                benchmark.download("https://unused.invalid", path, hashlib.sha256(b"valid").hexdigest())
            self.assertEqual(path.read_bytes(), b"valid")

    def test_report_aggregates_both_models(self):
        rows = [{"source": "synthetic.ARW", "model": name, "iso": 200,
                 "metrics": {"PSNR": 40, "MS-SSIM": 0.99, "HFEN": 0.1, "Mean ΔE00": 1}}
                for name in ("Okeinla", "RawNIND")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            benchmark.write_report(path, rows, {"limit": 1})
            text = path.read_text()
            self.assertIn("| Okeinla | 1 | 40.000000 |", text)
            self.assertIn("| RawNIND | 1 | 40.000000 |", text)
            self.assertIn('"limit": 1', text)
            self.assertIn("Subset run", text)

    def test_main_resumes_without_decoding_or_scoring_again(self):
        # Exercise the orchestration with mocked I/O, not actual model inference.
        class MetadataSession:
            def get_inputs(self):
                return [types.SimpleNamespace(name="input", shape=[1, 4, 512, 512], type="tensor(float)")]

            def get_outputs(self):
                return [types.SimpleNamespace(shape=[1, 3, 1024, 1024], type="tensor(float)")]

        source = "Scene_0001/Scene_0001_ISO200_1.ARW"
        reference = "Scene_0001/gt/Scene_0001_ISO100_2.ARW"
        packed = np.full((4, 16, 16), 0.2, dtype=np.float32)
        rgb = np.full((3, 32, 32), 0.2, dtype=np.float32)
        from benchmark_alignment import RgbAlignment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"synthetic metadata only")
            argv = ["benchmark.py", "--okeinla", str(model), "--rawnind", str(model),
                    "--output", str(root / "results"), "--limit", "1"]
            with (patch("sys.argv", argv),
                  patch.dict("sys.modules", {"onnxruntime":
                        types.SimpleNamespace(InferenceSession=lambda *a, **k: MetadataSession())}),
                  patch.object(benchmark, "dataset_manifest", return_value=(
                      {source: ("0" * 64, 10), reference: ("1" * 64, 10)}, [(source, reference)])),
                  patch.object(benchmark.importlib.metadata, "version", return_value="synthetic"),
                  patch.object(benchmark, "download", return_value=root / "unused"),
                  patch.object(benchmark, "decode", return_value=(packed, None)) as decode,
                  patch.object(benchmark, "demosaic", return_value=rgb),
                  patch.object(benchmark, "convert_color", return_value=rgb),
                  patch.object(benchmark, "estimate_rgb_alignment", return_value=RgbAlignment(0, 0, 1, 0, 1)),
                  patch.object(benchmark, "infer", return_value=rgb) as infer,
                  patch.object(benchmark, "metrics", return_value={
                      "PSNR": 40, "MS-SSIM": 0.99, "HFEN": 0.1, "Mean ΔE00": 1})):
                benchmark.main()
                self.assertEqual(infer.call_count, 2)
                decode.reset_mock()
                infer.reset_mock()
                benchmark.main()
                decode.assert_not_called()
                infer.assert_not_called()
            self.assertTrue((root / "results/report.md").is_file())


if __name__ == "__main__":
    unittest.main()
