"""Unit tests for the deterministic FWHT PSD codec."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from fwht_codec import (
    DatasetConfig,
    EncodedFWHTPayload,
    FWHTCodecConfig,
    PsdFrame,
    QuantizationState,
    compute_block_layout,
    compute_frame_metrics,
    compute_standardization_stats,
    decode_fwht_frame,
    decimate_psd,
    encode_fwht_frame,
    estimate_payload_bits,
    fwht_orthonormal,
    make_frequency_axis_hz,
    materialize_sparse_coefficients,
    total_occupied_bandwidth_hz,
)


def make_synthetic_frame(
    psd_db: np.ndarray,  # Synthetic PSD values on a uniform grid [dB]
) -> PsdFrame:
    """Build a minimal PSD frame suitable for codec and metric tests."""
    frequencies_hz = make_frequency_axis_hz(88e6, 108e6, psd_db.size)
    return PsdFrame(
        source_name="synthetic",
        psd_db=psd_db.astype(np.float64),
        frequencies_hz=frequencies_hz,
        timestamp_ms=0,
        start_freq_hz=88e6,
        end_freq_hz=108e6,
    )


class FWHTKernelTests(unittest.TestCase):
    """Tests for the mathematical FWHT kernel."""

    def test_fwht_is_orthonormal_and_self_inverse(self) -> None:
        """Applying the orthonormal FWHT twice must recover the input."""
        values = np.array([0.5, -1.0, 2.0, 3.5, -0.75, 1.25, 0.0, -2.5])
        transformed = fwht_orthonormal(values)
        recovered = fwht_orthonormal(transformed)

        np.testing.assert_allclose(recovered, values, atol=1e-12)
        self.assertAlmostEqual(
            float(np.linalg.norm(transformed)),
            float(np.linalg.norm(values)),
            places=12,
        )


class FWHTPayloadTests(unittest.TestCase):
    """Tests for payload-only reconstruction and rate accounting."""

    def test_payload_roundtrip_uses_quantized_side_information(self) -> None:
        """The payload must reconstruct only from transmitted indices, levels, and float32 side info."""
        frame = make_synthetic_frame(
            np.array(
                [
                    -70.0,
                    -69.5,
                    -68.0,
                    -40.0,
                    -39.0,
                    -67.5,
                    -68.5,
                    -69.0,
                    -70.5,
                    -69.8,
                    -68.7,
                ]
            )
        )
        codec_config = FWHTCodecConfig(
            decimation_factor=3,
            retained_coefficients=4,
            quantization_bits=7,
            aggregation_domain="db",
        )

        payload, diagnostics = encode_fwht_frame(frame, codec_config)
        reconstructed_psd_db = decode_fwht_frame(payload, codec_config)

        decimated_psd_db = decimate_psd(
            frame.psd_db,
            factor=codec_config.decimation_factor,
            aggregation_domain=codec_config.aggregation_domain,
        )
        expected_stats = compute_standardization_stats(
            decimated_psd_db,
            original_length=frame.psd_db.size,
        )
        expected_scale = float(
            np.float32(
                np.max(np.abs(diagnostics.dense_coefficients[payload.retained_indices]))
            )
        )

        self.assertEqual(payload.stats.mean_level, expected_stats.mean_level)
        self.assertEqual(payload.stats.std_level, expected_stats.std_level)
        self.assertEqual(payload.quantization.coefficient_scale, expected_scale)
        self.assertEqual(reconstructed_psd_db.shape[0], frame.psd_db.size)

    def test_decoder_rejects_duplicate_indices(self) -> None:
        """The payload validator must reject duplicated retained coefficient indices."""
        frame = make_synthetic_frame(np.linspace(-75.0, -35.0, 8))
        codec_config = FWHTCodecConfig(retained_coefficients=3, quantization_bits=6)
        payload, _ = encode_fwht_frame(frame, codec_config)
        bad_payload = EncodedFWHTPayload(
            retained_indices=np.array([1, 1, 3], dtype=np.int32),
            quantization=QuantizationState(
                quantized_levels=np.array([1, 2, 3], dtype=np.int32),
                coefficient_scale=payload.quantization.coefficient_scale,
            ),
            stats=payload.stats,
        )

        with self.assertRaises(ValueError):
            materialize_sparse_coefficients(bad_payload, codec_config)

    def test_payload_bit_estimate_matches_the_codec_contract(self) -> None:
        """Rate accounting must include side information and bit-packed retained indices."""
        frame = make_synthetic_frame(np.linspace(-80.0, -30.0, 16))
        codec_config = FWHTCodecConfig(
            decimation_factor=2,
            retained_coefficients=4,
            quantization_bits=8,
            aggregation_domain="db",
        )
        payload, _ = encode_fwht_frame(frame, codec_config)

        index_bits = 4 * 3
        value_bits = 4 * 8
        side_bits = 3 * 16 + 2 * 32 + 32
        self.assertEqual(
            estimate_payload_bits(payload, codec_config),
            index_bits + value_bits + side_bits,
        )


class SensingMetricTests(unittest.TestCase):
    """Tests for occupancy and connected-component metrics."""

    def test_occupancy_metric_uses_independent_noise_floor_estimates(self) -> None:
        """A constant dB shift should not change occupancy F1 when the same rule is applied independently."""
        reference_psd_db = np.array(
            [-80.0, -80.0, -60.0, -58.0, -80.0, -80.0, -79.0, -80.0]
        )
        reconstructed_psd_db = reference_psd_db + 10.0
        frame = make_synthetic_frame(reference_psd_db)
        dataset_config = DatasetConfig(dataset_dir=Path("."))

        metrics = compute_frame_metrics(
            frame=frame,
            reconstructed_psd_db=reconstructed_psd_db,
            dataset_config=dataset_config,
            payload_bits=128,
            codec_config=FWHTCodecConfig(),
        )

        self.assertAlmostEqual(metrics.occupancy_f1, 1.0, places=12)

    def test_connected_component_bandwidth_sums_each_region_width(self) -> None:
        """Bandwidth must be the sum of connected occupied widths, not the span over the full band."""
        frequencies_hz = np.arange(10, dtype=np.float64) + 0.5
        occupancy = np.array(
            [False, True, True, False, False, False, True, True, False, False]
        )

        self.assertEqual(total_occupied_bandwidth_hz(frequencies_hz, occupancy), 4.0)

    def test_block_layout_matches_the_last_partial_decimation_block(self) -> None:
        """The matched resampling geometry must preserve the shorter last block."""
        block_starts, block_lengths = compute_block_layout(num_bins=11, factor=3)

        np.testing.assert_array_equal(block_starts, np.array([0, 3, 6, 9]))
        np.testing.assert_array_equal(block_lengths, np.array([3, 3, 3, 2]))


if __name__ == "__main__":
    unittest.main()
