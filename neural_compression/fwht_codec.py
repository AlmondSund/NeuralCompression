"""FWHT codec core and sensing metrics for PSD compression experiments.

This module keeps the deterministic codec outside the notebook so the
implementation can be tested and reasoned about as a normal Python module.
The repository stores PSD samples in dB, so the codec accepts dB-domain PSD
frames as input. Decimation can still average in linear power before mapping
back to dB, which avoids the worst physical mismatch of directly averaging dB
values while preserving a stable dB-domain transform input for the FWHT stage.
"""

from __future__ import annotations

import ast
import csv
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

FLOAT32_BITS = 32
LENGTH_BITS = 16
MIN_LINEAR_POWER = 1e-30
SERIALIZATION_MAGIC = b"FWHT"
SERIALIZATION_VERSION = 1
SERIALIZATION_FLAG_HAS_SCALE = 1 << 0
MAX_LENGTH_VALUE = (1 << LENGTH_BITS) - 1
HEADER_FORMAT = "<4sBBBBBHHHHHff"
HEADER_SIZE_BYTES = struct.calcsize(HEADER_FORMAT)
NONLINEAR_MAP_CODES = {"identity": 0, "asinh": 1}
AGGREGATION_DOMAIN_CODES = {"db": 0, "linear_power": 1}
NONLINEAR_MAP_FROM_CODE = {code: name for name, code in NONLINEAR_MAP_CODES.items()}
AGGREGATION_DOMAIN_FROM_CODE = {
    code: name for name, code in AGGREGATION_DOMAIN_CODES.items()
}


@dataclass(frozen=True)
class PsdFrame:
    """A single PSD acquisition frame on a uniform frequency grid."""

    source_name: str
    psd_db: np.ndarray
    frequencies_hz: np.ndarray
    timestamp_ms: int
    start_freq_hz: float
    end_freq_hz: float

    def __post_init__(self) -> None:
        """Validate the frame assumptions required by the codec and metrics."""
        if self.psd_db.ndim != 1:
            raise ValueError("psd_db must be a one-dimensional array.")
        if self.frequencies_hz.shape != self.psd_db.shape:
            raise ValueError("frequencies_hz must match the PSD shape.")
        if self.psd_db.size == 0:
            raise ValueError("PSD frames must contain at least one bin.")
        if not np.all(np.isfinite(self.psd_db)):
            raise ValueError("psd_db must contain only finite values.")
        if not np.all(np.diff(self.frequencies_hz) > 0.0):
            raise ValueError("frequencies_hz must be strictly increasing.")


@dataclass(frozen=True)
class DatasetConfig:
    """Parameters that control data loading and sensing-level evaluation."""

    dataset_dir: Path
    max_frames: int | None = None
    noise_floor_percentile: float = 20.0
    occupancy_margin_db: float = 3.0

    def __post_init__(self) -> None:
        """Reject invalid dataset-level evaluation parameters."""
        if not 0.0 < self.noise_floor_percentile < 100.0:
            raise ValueError("noise_floor_percentile must lie in (0, 100).")
        if self.occupancy_margin_db < 0.0:
            raise ValueError("occupancy_margin_db must be non-negative.")


@dataclass(frozen=True)
class FWHTCodecConfig:
    """Configuration of the deterministic FWHT transform coder."""

    decimation_factor: int = 2
    retained_coefficients: int = 64
    quantization_bits: int = 8
    nonlinear_map: str = "identity"
    aggregation_domain: str = "linear_power"
    input_bits_per_bin: int = 32

    def __post_init__(self) -> None:
        """Validate codec hyperparameters that define the operating point."""
        if self.decimation_factor < 1:
            raise ValueError("decimation_factor must be at least one.")
        if self.decimation_factor > MAX_LENGTH_VALUE:
            raise ValueError(f"decimation_factor must fit inside {LENGTH_BITS} bits.")
        if self.retained_coefficients < 0:
            raise ValueError("retained_coefficients must be non-negative.")
        if self.quantization_bits < 1:
            raise ValueError("quantization_bits must be at least one.")
        if self.quantization_bits > 32:
            raise ValueError("quantization_bits must not exceed 32.")
        if self.nonlinear_map not in {"identity", "asinh"}:
            raise ValueError("nonlinear_map must be 'identity' or 'asinh'.")
        if self.aggregation_domain not in {"db", "linear_power"}:
            raise ValueError("aggregation_domain must be 'db' or 'linear_power'.")
        if self.input_bits_per_bin < 1:
            raise ValueError("input_bits_per_bin must be positive.")


@dataclass(frozen=True)
class StandardizationStats:
    """Per-frame side information required to invert standardization and resizing."""

    mean_level: float
    std_level: float
    original_length: int
    decimated_length: int
    padded_length: int


@dataclass(frozen=True)
class QuantizationState:
    """Metadata required to dequantize the retained FWHT coefficients."""

    quantized_levels: np.ndarray
    coefficient_scale: float


@dataclass(frozen=True)
class EncodedFWHTPayload:
    """Codec payload reconstructed only from transmitted indices, levels, and side information."""

    retained_indices: np.ndarray
    quantization: QuantizationState
    stats: StandardizationStats


@dataclass(frozen=True)
class FWHTEncodeDiagnostics:
    """Optional encoder-side diagnostics used for plots and analysis only."""

    dense_coefficients: np.ndarray


@dataclass(frozen=True)
class OccupancyComponent:
    """A connected occupied frequency interval derived from a binary occupancy mask."""

    start_index: int
    stop_index: int
    start_freq_hz: float
    stop_freq_hz: float
    bandwidth_hz: float
    center_freq_hz: float


@dataclass(frozen=True)
class FrameMetrics:
    """Waveform- and sensing-level metrics for a reconstructed PSD frame."""

    rmse_db: float
    nmse: float
    occupancy_f1: float
    occupancy_false_negative_rate: float
    occupancy_false_positive_rate: float
    peak_error_hz: float
    centroid_error_hz: float
    occupied_bandwidth_error_hz: float
    payload_bits: int
    compression_ratio: float


def make_frequency_axis_hz(
    start_freq_hz: float,  # Lower edge of the monitored band [Hz]
    end_freq_hz: float,  # Upper edge of the monitored band [Hz]
    num_bins: int,  # Number of PSD bins
) -> np.ndarray:
    """Return the center frequency of each uniform PSD bin."""
    if num_bins < 1:
        raise ValueError("num_bins must be positive.")

    bin_width_hz = (end_freq_hz - start_freq_hz) / num_bins
    return start_freq_hz + bin_width_hz * (0.5 + np.arange(num_bins, dtype=np.float64))


def parse_psd_values(
    values_literal: str,  # Serialized Python list of PSD values in dB
) -> np.ndarray:
    """Parse the CSV payload into a finite one-dimensional float array."""
    values = np.asarray(ast.literal_eval(values_literal), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("The PSD payload must decode to a one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("The PSD payload contains NaN or Inf values.")
    return values


def load_psd_frames(
    dataset_config: DatasetConfig,
) -> list[PsdFrame]:
    """Load PSD frames from every CSV file in the configured dataset directory."""
    frames: list[PsdFrame] = []

    # Sorting both files and rows makes the dataset traversal deterministic.
    for csv_path in sorted(dataset_config.dataset_dir.glob("*.csv")):
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                psd_db = parse_psd_values(row["pxx"])
                start_freq_hz = float(row["start_freq_hz"])
                end_freq_hz = float(row["end_freq_hz"])
                frequencies_hz = make_frequency_axis_hz(
                    start_freq_hz=start_freq_hz,
                    end_freq_hz=end_freq_hz,
                    num_bins=psd_db.size,
                )
                frames.append(
                    PsdFrame(
                        source_name=f"{csv_path.stem}:{row_index}",
                        psd_db=psd_db,
                        frequencies_hz=frequencies_hz,
                        timestamp_ms=int(row["timestamp"]),
                        start_freq_hz=start_freq_hz,
                        end_freq_hz=end_freq_hz,
                    )
                )
                if (
                    dataset_config.max_frames is not None
                    and len(frames) >= dataset_config.max_frames
                ):
                    return frames
    if not frames:
        raise FileNotFoundError(
            f"No PSD frames were found in {dataset_config.dataset_dir}."
        )
    return frames


def quantize_float32_scalar(
    value: float,  # Side-information scalar to be transmitted as float32
) -> float:
    """Quantize a scalar to float32 and return it as a Python float."""
    return float(np.float32(value))


def db_to_linear_power(
    values_db: np.ndarray,  # PSD or power values in dB
) -> np.ndarray:
    """Convert dB values into linear power while preserving array shape."""
    return np.power(10.0, values_db / 10.0, dtype=np.float64)


def linear_power_to_db(
    values_power: np.ndarray,  # PSD or power values in linear units
) -> np.ndarray:
    """Convert linear power into dB with an explicit lower power floor."""
    return 10.0 * np.log10(np.maximum(values_power, MIN_LINEAR_POWER))


def compute_block_layout(
    num_bins: int,  # Number of original PSD bins
    factor: int,  # Number of adjacent bins grouped into one block
) -> tuple[np.ndarray, np.ndarray]:
    """Return the start index and length of every decimation block."""
    if num_bins < 1:
        raise ValueError("num_bins must be positive.")
    if factor < 1:
        raise ValueError("factor must be at least one.")

    block_starts = np.arange(0, num_bins, factor, dtype=np.int64)
    block_lengths = np.minimum(
        np.full_like(block_starts, fill_value=factor),
        num_bins - block_starts,
    )
    return block_starts, block_lengths


def block_center_positions(
    num_bins: int,  # Number of bins on the original grid
    factor: int,  # Number of original bins represented by each decimated sample
) -> np.ndarray:
    """Return the center position of every decimation block on the original bin axis."""
    block_starts, block_lengths = compute_block_layout(num_bins, factor)
    return block_starts.astype(np.float64) + block_lengths.astype(np.float64) / 2.0


def decimate_psd(
    values_db: np.ndarray,  # PSD bins on a uniform grid [dB]
    factor: int,  # Number of adjacent bins averaged together
    aggregation_domain: str,  # Domain used to average adjacent bins
) -> np.ndarray:
    """Downsample a PSD by averaging adjacent bins in the requested domain."""
    if values_db.ndim != 1:
        raise ValueError("values_db must be one-dimensional.")
    if factor < 1:
        raise ValueError("factor must be at least one.")
    if factor == 1:
        return values_db.copy()

    block_starts, block_lengths = compute_block_layout(values_db.size, factor)
    if aggregation_domain == "linear_power":
        source_values = db_to_linear_power(values_db)
        block_sums = np.add.reduceat(source_values, block_starts)
        block_means = block_sums / block_lengths
        return linear_power_to_db(block_means)
    if aggregation_domain == "db":
        block_sums = np.add.reduceat(values_db, block_starts)
        return block_sums / block_lengths
    raise ValueError(f"Unsupported aggregation_domain: {aggregation_domain}")


def upsample_psd(
    values_db: np.ndarray,  # Decimated PSD bins [dB]
    target_length: int,  # Desired number of bins after interpolation
    factor: int,  # Number of original bins represented by each decimated value
    aggregation_domain: str,  # Domain used to interpolate the decimated values
) -> np.ndarray:
    """Interpolate a decimated PSD back onto the original uniform bin centers."""
    if values_db.ndim != 1:
        raise ValueError("values_db must be one-dimensional.")
    if target_length < 1:
        raise ValueError("target_length must be positive.")
    if factor < 1:
        raise ValueError("factor must be at least one.")
    if values_db.size == target_length:
        return values_db.copy()

    source_positions = block_center_positions(target_length, factor)
    if source_positions.size != values_db.size:
        raise ValueError(
            "values_db length does not match the decimation layout implied by target_length and factor."
        )
    target_positions = 0.5 + np.arange(target_length, dtype=np.float64)

    # Interpolation happens at block-center coordinates so the upsampler matches the decimator geometry.
    if aggregation_domain == "linear_power":
        source_values = db_to_linear_power(values_db)
        interpolated = np.interp(
            target_positions,
            source_positions,
            source_values,
            left=float(source_values[0]),
            right=float(source_values[-1]),
        )
        return linear_power_to_db(interpolated)
    if aggregation_domain == "db":
        return np.interp(
            target_positions,
            source_positions,
            values_db,
            left=float(values_db[0]),
            right=float(values_db[-1]),
        )
    raise ValueError(f"Unsupported aggregation_domain: {aggregation_domain}")


def compute_standardization_stats(
    values: np.ndarray,  # Decimated PSD bins in the transform input domain
    original_length: int,  # Number of bins on the original PSD grid
) -> StandardizationStats:
    """Compute float32 side information required to invert standardization and resizing."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if original_length < 1:
        raise ValueError("original_length must be positive.")

    mean_level = quantize_float32_scalar(float(np.mean(values)))
    std_level = quantize_float32_scalar(float(np.std(values)))
    if std_level < 1e-8:
        std_level = 1.0
    return StandardizationStats(
        mean_level=mean_level,
        std_level=std_level,
        original_length=original_length,
        decimated_length=values.size,
        padded_length=next_power_of_two(values.size),
    )


def standardize_values(
    values: np.ndarray,  # Decimated PSD bins in the transform input domain
    mean_level: float,  # Mean transmitted as side information
    std_level: float,  # Standard deviation transmitted as side information
) -> np.ndarray:
    """Apply z-score standardization using externally supplied statistics."""
    return (values - mean_level) / std_level


def destandardize_values(
    values: np.ndarray,  # Standardized values after inverse FWHT
    mean_level: float,  # Mean transmitted as side information
    std_level: float,  # Standard deviation transmitted as side information
) -> np.ndarray:
    """Undo z-score standardization using the transmitted side information."""
    return values * std_level + mean_level


def apply_nonlinear_map(
    values: np.ndarray,  # Standardized values before FWHT
    nonlinear_map: str,  # Name of the invertible pointwise map
) -> np.ndarray:
    """Apply the configured invertible pointwise map before the FWHT."""
    if nonlinear_map == "identity":
        return values
    if nonlinear_map == "asinh":
        return np.arcsinh(values)
    raise ValueError(f"Unsupported nonlinear_map: {nonlinear_map}")


def invert_nonlinear_map(
    values: np.ndarray,  # Inverse-FWHT output before destandardization
    nonlinear_map: str,  # Name of the invertible pointwise map
) -> np.ndarray:
    """Invert the configured pointwise map after the inverse FWHT."""
    if nonlinear_map == "identity":
        return values
    if nonlinear_map == "asinh":
        return np.sinh(values)
    raise ValueError(f"Unsupported nonlinear_map: {nonlinear_map}")


def next_power_of_two(
    value: int,  # Positive array length
) -> int:
    """Return the smallest power of two greater than or equal to value."""
    if value < 1:
        raise ValueError("value must be positive.")
    return 1 if value == 1 else 1 << (value - 1).bit_length()


def fwht_orthonormal(
    values: np.ndarray,  # Length must be a power of two
) -> np.ndarray:
    """Compute the orthonormal Fast Walsh-Hadamard Transform."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if values.size == 0 or values.size & (values.size - 1):
        raise ValueError("FWHT requires a non-empty power-of-two length.")

    transformed = np.asarray(values, dtype=np.float64).copy()
    butterfly_width = 1

    # Each stage applies the Sylvester-ordered Hadamard butterflies to consecutive blocks.
    while butterfly_width < transformed.size:
        reshaped = transformed.reshape(-1, 2 * butterfly_width)
        left_half = reshaped[:, :butterfly_width].copy()
        right_half = reshaped[:, butterfly_width:].copy()
        reshaped[:, :butterfly_width] = left_half + right_half
        reshaped[:, butterfly_width:] = left_half - right_half
        butterfly_width *= 2
    return transformed / math.sqrt(values.size)


def quantize_symmetric_uniform(
    values: np.ndarray,  # Retained Hadamard coefficients
    quantization_bits: int,  # Signed quantizer precision [bits]
) -> QuantizationState:
    """Quantize coefficients with a symmetric uniform mid-tread quantizer."""
    if quantization_bits < 1:
        raise ValueError("quantization_bits must be positive.")
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if values.size == 0:
        return QuantizationState(
            quantized_levels=np.zeros(0, dtype=np.int32),
            coefficient_scale=1.0,
        )

    quantization_levels = max(1, (1 << (quantization_bits - 1)) - 1)
    coefficient_scale = float(np.max(np.abs(values)))
    if coefficient_scale < 1e-12:
        quantized_levels = np.zeros(values.size, dtype=np.int32)
        coefficient_scale = 1.0
    else:
        normalized = values / coefficient_scale
        quantized_levels = np.clip(
            np.rint(normalized * quantization_levels),
            -quantization_levels,
            quantization_levels,
        ).astype(np.int32)
        coefficient_scale = quantize_float32_scalar(coefficient_scale)
    return QuantizationState(
        quantized_levels=quantized_levels,
        coefficient_scale=coefficient_scale,
    )


def dequantize_symmetric_uniform(
    quantization: QuantizationState,
    quantization_bits: int,
) -> np.ndarray:
    """Reconstruct retained coefficients from the quantized integer levels."""
    if quantization_bits < 1:
        raise ValueError("quantization_bits must be positive.")
    if quantization.quantized_levels.ndim != 1:
        raise ValueError("quantized_levels must be one-dimensional.")
    if quantization.quantized_levels.size == 0:
        return np.zeros(0, dtype=np.float64)

    quantization_levels = max(1, (1 << (quantization_bits - 1)) - 1)
    return (
        quantization.quantized_levels.astype(np.float64)
        / quantization_levels
        * quantization.coefficient_scale
    )


def validate_payload(
    payload: EncodedFWHTPayload,  # Candidate FWHT payload
    codec_config: FWHTCodecConfig,  # Codec contract used to interpret the payload
) -> None:
    """Validate the payload invariants required by the decoder."""
    if payload.retained_indices.ndim != 1:
        raise ValueError("retained_indices must be one-dimensional.")
    if not np.issubdtype(payload.retained_indices.dtype, np.integer):
        raise ValueError("retained_indices must use an integer dtype.")
    if payload.quantization.quantized_levels.ndim != 1:
        raise ValueError("quantized_levels must be one-dimensional.")
    if not np.issubdtype(payload.quantization.quantized_levels.dtype, np.integer):
        raise ValueError("quantized_levels must use an integer dtype.")
    if payload.retained_indices.size != payload.quantization.quantized_levels.size:
        raise ValueError(
            "retained_indices and quantized_levels must contain the same number of entries."
        )
    if payload.stats.original_length < 1:
        raise ValueError("original_length must be positive.")
    if payload.stats.original_length > MAX_LENGTH_VALUE:
        raise ValueError(f"original_length must fit inside {LENGTH_BITS} bits.")
    if payload.stats.decimated_length < 1:
        raise ValueError("decimated_length must be positive.")
    if payload.stats.decimated_length > MAX_LENGTH_VALUE:
        raise ValueError(f"decimated_length must fit inside {LENGTH_BITS} bits.")
    if payload.stats.padded_length < payload.stats.decimated_length:
        raise ValueError("padded_length must be at least decimated_length.")
    if payload.stats.padded_length > MAX_LENGTH_VALUE:
        raise ValueError(f"padded_length must fit inside {LENGTH_BITS} bits.")
    if payload.stats.padded_length & (payload.stats.padded_length - 1):
        raise ValueError("padded_length must be a power of two.")
    if not math.isfinite(payload.stats.mean_level):
        raise ValueError("mean_level must be finite.")
    if not math.isfinite(payload.stats.std_level) or payload.stats.std_level <= 0.0:
        raise ValueError("std_level must be finite and strictly positive.")
    if not math.isfinite(payload.quantization.coefficient_scale):
        raise ValueError("coefficient_scale must be finite.")
    if payload.quantization.coefficient_scale < 0.0:
        raise ValueError("coefficient_scale must be non-negative.")
    if payload.retained_indices.size > MAX_LENGTH_VALUE:
        raise ValueError(
            f"retained coefficient count must fit inside {LENGTH_BITS} bits."
        )
    if payload.retained_indices.size > payload.stats.padded_length:
        raise ValueError("retained coefficient count must not exceed padded_length.")

    expected_decimated_length = compute_block_layout(
        payload.stats.original_length,
        codec_config.decimation_factor,
    )[0].size
    if payload.stats.decimated_length != expected_decimated_length:
        raise ValueError(
            "decimated_length is inconsistent with original_length and decimation_factor."
        )
    if payload.stats.padded_length != next_power_of_two(payload.stats.decimated_length):
        raise ValueError("padded_length is inconsistent with decimated_length.")

    quantization_levels = max(1, (1 << (codec_config.quantization_bits - 1)) - 1)
    if np.any(np.abs(payload.quantization.quantized_levels) > quantization_levels):
        raise ValueError(
            "quantized_levels exceed the valid range for quantization_bits."
        )
    if payload.retained_indices.size == 0:
        return
    if payload.quantization.coefficient_scale <= 0.0:
        raise ValueError(
            "coefficient_scale must be strictly positive when retained coefficients are present."
        )

    if int(payload.retained_indices[0]) < 0:
        raise ValueError("retained_indices must be non-negative.")
    if int(payload.retained_indices[-1]) >= payload.stats.padded_length:
        raise ValueError(
            "retained_indices must lie inside the padded transform length."
        )
    if np.any(np.diff(payload.retained_indices) <= 0):
        raise ValueError("retained_indices must be strictly increasing and unique.")


def quantized_level_dtype(
    quantization_bits: int,  # Signed quantizer precision [bits]
) -> np.dtype:
    """Return the smallest signed integer dtype that can store the quantized levels."""
    if quantization_bits < 1 or quantization_bits > 32:
        raise ValueError("quantization_bits must lie in [1, 32].")
    if quantization_bits <= 8:
        return np.dtype("<i1")
    if quantization_bits <= 16:
        return np.dtype("<i2")
    return np.dtype("<i4")


def serialize_payload(
    payload: EncodedFWHTPayload,  # Payload to serialize
    codec_config: FWHTCodecConfig,  # Codec contract that defines field widths
) -> bytes:
    """Serialize a validated FWHT payload into a versioned little-endian byte packet."""
    validate_payload(payload, codec_config)

    retained_count = payload.retained_indices.size
    has_scale = retained_count > 0
    flags = SERIALIZATION_FLAG_HAS_SCALE if has_scale else 0
    header_bytes = struct.pack(
        HEADER_FORMAT,
        SERIALIZATION_MAGIC,
        SERIALIZATION_VERSION,
        flags,
        NONLINEAR_MAP_CODES[codec_config.nonlinear_map],
        AGGREGATION_DOMAIN_CODES[codec_config.aggregation_domain],
        codec_config.quantization_bits,
        codec_config.decimation_factor,
        payload.stats.original_length,
        payload.stats.decimated_length,
        payload.stats.padded_length,
        retained_count,
        payload.stats.mean_level,
        payload.stats.std_level,
    )

    scale_bytes = b""
    if has_scale:
        scale_bytes = struct.pack("<f", payload.quantization.coefficient_scale)

    index_bytes = payload.retained_indices.astype("<u2", copy=False).tobytes()
    level_dtype = quantized_level_dtype(codec_config.quantization_bits)
    level_bytes = payload.quantization.quantized_levels.astype(
        level_dtype,
        copy=False,
    ).tobytes()
    return header_bytes + scale_bytes + index_bytes + level_bytes


def deserialize_payload(
    blob: bytes | bytearray | memoryview,  # Serialized FWHT packet
    codec_config: FWHTCodecConfig,  # Codec contract expected by the receiver
) -> EncodedFWHTPayload:
    """Deserialize a versioned FWHT packet into a validated payload object."""
    packet = bytes(blob)
    if len(packet) < HEADER_SIZE_BYTES:
        raise ValueError("Serialized payload is shorter than the FWHT header.")

    (
        magic,
        version,
        flags,
        nonlinear_code,
        aggregation_code,
        quantization_bits,
        decimation_factor,
        original_length,
        decimated_length,
        padded_length,
        retained_count,
        mean_level,
        std_level,
    ) = struct.unpack_from(HEADER_FORMAT, packet, 0)
    if magic != SERIALIZATION_MAGIC:
        raise ValueError("Serialized payload has an invalid magic prefix.")
    if version != SERIALIZATION_VERSION:
        raise ValueError("Serialized payload uses an unsupported version.")
    if flags & ~SERIALIZATION_FLAG_HAS_SCALE:
        raise ValueError("Serialized payload uses unsupported header flags.")
    if nonlinear_code not in NONLINEAR_MAP_FROM_CODE:
        raise ValueError("Serialized payload uses an unknown nonlinear map code.")
    if aggregation_code not in AGGREGATION_DOMAIN_FROM_CODE:
        raise ValueError("Serialized payload uses an unknown aggregation domain code.")
    if NONLINEAR_MAP_FROM_CODE[nonlinear_code] != codec_config.nonlinear_map:
        raise ValueError(
            "Serialized payload nonlinear map does not match codec_config."
        )
    if (
        AGGREGATION_DOMAIN_FROM_CODE[aggregation_code]
        != codec_config.aggregation_domain
    ):
        raise ValueError(
            "Serialized payload aggregation domain does not match codec_config."
        )
    if quantization_bits != codec_config.quantization_bits:
        raise ValueError(
            "Serialized payload quantization_bits does not match codec_config."
        )
    if decimation_factor != codec_config.decimation_factor:
        raise ValueError(
            "Serialized payload decimation_factor does not match codec_config."
        )

    offset = HEADER_SIZE_BYTES
    coefficient_scale = 0.0
    if flags & SERIALIZATION_FLAG_HAS_SCALE:
        scale_end = offset + FLOAT32_BITS // 8
        if len(packet) < scale_end:
            raise ValueError(
                "Serialized payload is truncated before coefficient_scale."
            )
        (coefficient_scale,) = struct.unpack_from("<f", packet, offset)
        offset = scale_end
    elif retained_count > 0:
        raise ValueError(
            "Serialized payload omitted coefficient_scale despite carrying retained coefficients."
        )

    index_end = offset + retained_count * (LENGTH_BITS // 8)
    if len(packet) < index_end:
        raise ValueError("Serialized payload is truncated before retained_indices.")
    retained_indices = np.frombuffer(
        packet[offset:index_end],
        dtype="<u2",
    ).astype(np.int32)
    offset = index_end

    level_dtype = quantized_level_dtype(codec_config.quantization_bits)
    level_end = offset + retained_count * level_dtype.itemsize
    if len(packet) != level_end:
        raise ValueError(
            "Serialized payload length is inconsistent with the header metadata."
        )
    quantized_levels = np.frombuffer(
        packet[offset:level_end],
        dtype=level_dtype,
    ).astype(np.int32)

    payload = EncodedFWHTPayload(
        retained_indices=retained_indices,
        quantization=QuantizationState(
            quantized_levels=quantized_levels,
            coefficient_scale=float(coefficient_scale),
        ),
        stats=StandardizationStats(
            mean_level=float(mean_level),
            std_level=float(std_level),
            original_length=int(original_length),
            decimated_length=int(decimated_length),
            padded_length=int(padded_length),
        ),
    )
    validate_payload(payload, codec_config)
    return payload


def materialize_sparse_coefficients(
    payload: EncodedFWHTPayload,
    codec_config: FWHTCodecConfig,
) -> np.ndarray:
    """Rebuild the sparse Hadamard-domain vector from the transmitted payload only."""
    validate_payload(payload, codec_config)

    sparse_coefficients = np.zeros(payload.stats.padded_length, dtype=np.float64)
    if payload.retained_indices.size == 0:
        return sparse_coefficients

    retained_values = dequantize_symmetric_uniform(
        payload.quantization,
        codec_config.quantization_bits,
    )
    sparse_coefficients[payload.retained_indices] = retained_values
    return sparse_coefficients


def encode_fwht_frame(
    frame: PsdFrame,
    codec_config: FWHTCodecConfig,
) -> tuple[EncodedFWHTPayload, FWHTEncodeDiagnostics]:
    """Encode a PSD frame with decimation, FWHT sparsification, and scalar quantization."""
    decimated_psd_db = decimate_psd(
        frame.psd_db,
        factor=codec_config.decimation_factor,
        aggregation_domain=codec_config.aggregation_domain,
    )
    stats = compute_standardization_stats(
        decimated_psd_db,
        original_length=frame.psd_db.size,
    )
    standardized = standardize_values(
        decimated_psd_db,
        mean_level=stats.mean_level,
        std_level=stats.std_level,
    )
    transformed_signal = apply_nonlinear_map(
        standardized,
        codec_config.nonlinear_map,
    )

    # Zero-padding matches the orthonormal Hadamard matrix size required by the FWHT.
    padded_signal = np.pad(
        transformed_signal,
        (0, stats.padded_length - transformed_signal.size),
        mode="constant",
        constant_values=0.0,
    )
    dense_coefficients = fwht_orthonormal(padded_signal)

    retained_coefficients = min(
        codec_config.retained_coefficients,
        dense_coefficients.size,
    )
    if retained_coefficients == 0:
        retained_indices = np.zeros(0, dtype=np.int32)
        quantization = quantize_symmetric_uniform(
            np.zeros(0, dtype=np.float64),
            codec_config.quantization_bits,
        )
    else:
        retained_index_candidates = np.argpartition(
            np.abs(dense_coefficients),
            -retained_coefficients,
        )[-retained_coefficients:]
        retained_indices = np.sort(retained_index_candidates.astype(np.int32))
        quantization = quantize_symmetric_uniform(
            dense_coefficients[retained_indices],
            codec_config.quantization_bits,
        )

    payload = EncodedFWHTPayload(
        retained_indices=retained_indices,
        quantization=quantization,
        stats=stats,
    )
    diagnostics = FWHTEncodeDiagnostics(dense_coefficients=dense_coefficients)
    return payload, diagnostics


def decode_fwht_frame(
    payload: EncodedFWHTPayload | bytes | bytearray | memoryview,
    codec_config: FWHTCodecConfig,
) -> np.ndarray:
    """Decode an FWHT payload back onto the original PSD grid."""
    if isinstance(payload, (bytes, bytearray, memoryview)):
        payload = deserialize_payload(payload, codec_config)

    sparse_coefficients = materialize_sparse_coefficients(payload, codec_config)
    transformed_signal = fwht_orthonormal(sparse_coefficients)[
        : payload.stats.decimated_length
    ]
    standardized = invert_nonlinear_map(
        transformed_signal,
        codec_config.nonlinear_map,
    )
    decimated_psd_db = destandardize_values(
        standardized,
        mean_level=payload.stats.mean_level,
        std_level=payload.stats.std_level,
    )
    return upsample_psd(
        decimated_psd_db,
        target_length=payload.stats.original_length,
        factor=codec_config.decimation_factor,
        aggregation_domain=codec_config.aggregation_domain,
    )


def estimate_payload_bits(
    payload: EncodedFWHTPayload,
    codec_config: FWHTCodecConfig,
) -> int:
    """Estimate the transmitted payload bits for the current encoded frame."""
    return len(serialize_payload(payload, codec_config)) * 8


def reconstruct_fwht_frame(
    frame: PsdFrame,
    codec_config: FWHTCodecConfig,
) -> tuple[EncodedFWHTPayload, FWHTEncodeDiagnostics, np.ndarray]:
    """Run the full FWHT encode/decode loop for one frame."""
    payload, diagnostics = encode_fwht_frame(frame, codec_config)
    payload_bytes = serialize_payload(payload, codec_config)
    reconstructed_psd_db = decode_fwht_frame(payload_bytes, codec_config)
    return payload, diagnostics, reconstructed_psd_db


def estimate_noise_floor_db(
    psd_db: np.ndarray,
    percentile: float,
) -> float:
    """Estimate a scalar noise floor from a low PSD percentile in dB."""
    return float(np.percentile(psd_db, percentile))


def occupancy_mask(
    psd_db: np.ndarray,
    threshold_db: float,
) -> np.ndarray:
    """Convert a PSD into a binary occupancy mask using a scalar threshold."""
    return psd_db > threshold_db


def spectral_peak_frequency_hz(
    frame: PsdFrame,
    psd_db: np.ndarray,
) -> float:
    """Return the frequency of the strongest PSD bin."""
    return float(frame.frequencies_hz[int(np.argmax(psd_db))])


def spectral_centroid_hz(
    frame: PsdFrame,
    psd_db: np.ndarray,
) -> float:
    """Return the power-weighted spectral centroid using a stable dB-to-linear conversion."""
    relative_linear_power = db_to_linear_power(psd_db - float(np.max(psd_db)))
    total_power = float(np.sum(relative_linear_power))
    if total_power <= 0.0:
        return float(np.mean(frame.frequencies_hz))
    return float(np.sum(frame.frequencies_hz * relative_linear_power) / total_power)


def extract_occupied_components(
    frequencies_hz: np.ndarray,
    occupancy: np.ndarray,
) -> list[OccupancyComponent]:
    """Extract connected occupied frequency intervals from a binary occupancy mask."""
    if frequencies_hz.ndim != 1 or occupancy.ndim != 1:
        raise ValueError("frequencies_hz and occupancy must be one-dimensional.")
    if frequencies_hz.shape != occupancy.shape:
        raise ValueError("frequencies_hz and occupancy must have the same shape.")
    if occupancy.size == 0:
        return []

    occupied_indices = np.flatnonzero(occupancy)
    if occupied_indices.size == 0:
        return []

    bin_width_hz = (
        0.0 if frequencies_hz.size < 2 else float(np.mean(np.diff(frequencies_hz)))
    )
    split_points = np.where(np.diff(occupied_indices) > 1)[0] + 1
    components: list[OccupancyComponent] = []

    # Connected components match the writeup better than a single span from the first to last occupied bin.
    for component_indices in np.split(occupied_indices, split_points):
        start_index = int(component_indices[0])
        stop_index = int(component_indices[-1]) + 1
        start_freq_hz = float(frequencies_hz[start_index] - bin_width_hz / 2.0)
        stop_freq_hz = float(frequencies_hz[stop_index - 1] + bin_width_hz / 2.0)
        bandwidth_hz = stop_freq_hz - start_freq_hz
        components.append(
            OccupancyComponent(
                start_index=start_index,
                stop_index=stop_index,
                start_freq_hz=start_freq_hz,
                stop_freq_hz=stop_freq_hz,
                bandwidth_hz=bandwidth_hz,
                center_freq_hz=0.5 * (start_freq_hz + stop_freq_hz),
            )
        )
    return components


def total_occupied_bandwidth_hz(
    frequencies_hz: np.ndarray,
    occupancy: np.ndarray,
) -> float:
    """Return the total occupied support width after connected-component extraction."""
    return float(
        sum(
            component.bandwidth_hz
            for component in extract_occupied_components(frequencies_hz, occupancy)
        )
    )


def compute_frame_metrics(
    frame: PsdFrame,
    reconstructed_psd_db: np.ndarray,
    dataset_config: DatasetConfig,
    payload_bits: int,
    codec_config: FWHTCodecConfig,
) -> FrameMetrics:
    """Compute waveform fidelity and sensing-level errors for one reconstructed frame."""
    residual = reconstructed_psd_db - frame.psd_db
    residual_norm_sq = float(np.linalg.norm(residual) ** 2)
    reference_norm_sq = float(np.linalg.norm(frame.psd_db) ** 2)
    rmse_db = float(np.sqrt(np.mean(residual**2)))
    nmse = 0.0 if reference_norm_sq <= 1e-12 else residual_norm_sq / reference_norm_sq

    reference_noise_floor_db = estimate_noise_floor_db(
        frame.psd_db,
        dataset_config.noise_floor_percentile,
    )
    reconstructed_noise_floor_db = estimate_noise_floor_db(
        reconstructed_psd_db,
        dataset_config.noise_floor_percentile,
    )
    reference_threshold_db = (
        reference_noise_floor_db + dataset_config.occupancy_margin_db
    )
    reconstructed_threshold_db = (
        reconstructed_noise_floor_db + dataset_config.occupancy_margin_db
    )
    reference_occupancy = occupancy_mask(frame.psd_db, reference_threshold_db)
    reconstructed_occupancy = occupancy_mask(
        reconstructed_psd_db,
        reconstructed_threshold_db,
    )

    tp = int(np.count_nonzero(reference_occupancy & reconstructed_occupancy))
    tn = int(np.count_nonzero(~reference_occupancy & ~reconstructed_occupancy))
    fp = int(np.count_nonzero(~reference_occupancy & reconstructed_occupancy))
    fn = int(np.count_nonzero(reference_occupancy & ~reconstructed_occupancy))

    occupancy_denominator = 2 * tp + fp + fn
    occupancy_f1 = (
        1.0 if occupancy_denominator == 0 else (2.0 * tp) / occupancy_denominator
    )
    occupancy_false_negative_rate = 0.0 if (tp + fn) == 0 else fn / (tp + fn)
    occupancy_false_positive_rate = 0.0 if (fp + tn) == 0 else fp / (fp + tn)

    peak_error_hz = abs(
        spectral_peak_frequency_hz(frame, frame.psd_db)
        - spectral_peak_frequency_hz(frame, reconstructed_psd_db)
    )
    centroid_error_hz = abs(
        spectral_centroid_hz(frame, frame.psd_db)
        - spectral_centroid_hz(frame, reconstructed_psd_db)
    )
    occupied_bandwidth_error_hz = abs(
        total_occupied_bandwidth_hz(frame.frequencies_hz, reference_occupancy)
        - total_occupied_bandwidth_hz(
            frame.frequencies_hz,
            reconstructed_occupancy,
        )
    )

    original_payload_bits = frame.psd_db.size * codec_config.input_bits_per_bin
    compression_ratio = float(original_payload_bits / payload_bits)

    return FrameMetrics(
        rmse_db=rmse_db,
        nmse=nmse,
        occupancy_f1=float(occupancy_f1),
        occupancy_false_negative_rate=float(occupancy_false_negative_rate),
        occupancy_false_positive_rate=float(occupancy_false_positive_rate),
        peak_error_hz=float(peak_error_hz),
        centroid_error_hz=float(centroid_error_hz),
        occupied_bandwidth_error_hz=float(occupied_bandwidth_error_hz),
        payload_bits=int(payload_bits),
        compression_ratio=compression_ratio,
    )


def evaluate_codec_dataset(
    frames: list[PsdFrame],
    dataset_config: DatasetConfig,
    codec_configs: list[FWHTCodecConfig],
) -> pd.DataFrame:
    """Evaluate several FWHT operating points over a dataset of PSD frames."""
    records: list[dict[str, float | int | str]] = []
    for codec_config in codec_configs:
        for frame in frames:
            payload, _diagnostics, reconstructed_psd_db = reconstruct_fwht_frame(
                frame,
                codec_config,
            )
            payload_bits = estimate_payload_bits(payload, codec_config)
            frame_metrics = compute_frame_metrics(
                frame=frame,
                reconstructed_psd_db=reconstructed_psd_db,
                dataset_config=dataset_config,
                payload_bits=payload_bits,
                codec_config=codec_config,
            )
            record = asdict(frame_metrics)
            record.update(
                {
                    "source_name": frame.source_name,
                    "decimation_factor": codec_config.decimation_factor,
                    "retained_coefficients": codec_config.retained_coefficients,
                    "quantization_bits": codec_config.quantization_bits,
                    "nonlinear_map": codec_config.nonlinear_map,
                    "aggregation_domain": codec_config.aggregation_domain,
                }
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_results(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate frame-level metrics into one row per FWHT operating point."""
    summary_df = (
        results_df.groupby(
            [
                "decimation_factor",
                "nonlinear_map",
                "aggregation_domain",
                "quantization_bits",
                "retained_coefficients",
            ],
            as_index=False,
        )
        .agg(
            mean_rmse_db=("rmse_db", "mean"),
            mean_nmse=("nmse", "mean"),
            mean_occupancy_f1=("occupancy_f1", "mean"),
            mean_peak_error_hz=("peak_error_hz", "mean"),
            mean_centroid_error_hz=("centroid_error_hz", "mean"),
            mean_bandwidth_error_hz=("occupied_bandwidth_error_hz", "mean"),
            mean_payload_bits=("payload_bits", "mean"),
            mean_compression_ratio=("compression_ratio", "mean"),
        )
        .sort_values(
            [
                "decimation_factor",
                "nonlinear_map",
                "aggregation_domain",
                "retained_coefficients",
            ]
        )
        .reset_index(drop=True)
    )
    summary_df["mean_payload_bytes"] = summary_df["mean_payload_bits"] / 8.0
    return summary_df


def select_representative_frame(
    frames: list[PsdFrame],
    dataset_config: DatasetConfig,
) -> PsdFrame:
    """Pick a frame with a strong emission to make reconstruction plots informative."""
    best_score = -np.inf
    best_frame = frames[0]
    for frame in frames:
        noise_floor_db = estimate_noise_floor_db(
            frame.psd_db,
            dataset_config.noise_floor_percentile,
        )
        occupancy = occupancy_mask(
            frame.psd_db,
            noise_floor_db + dataset_config.occupancy_margin_db,
        )
        score = float(np.max(frame.psd_db) - noise_floor_db) + 0.1 * float(
            np.count_nonzero(occupancy)
        )
        if score > best_score:
            best_score = score
            best_frame = frame
    return best_frame


__all__ = [
    "AGGREGATION_DOMAIN_CODES",
    "AGGREGATION_DOMAIN_FROM_CODE",
    "DatasetConfig",
    "EncodedFWHTPayload",
    "FWHTCodecConfig",
    "FWHTEncodeDiagnostics",
    "FLOAT32_BITS",
    "FrameMetrics",
    "HEADER_FORMAT",
    "HEADER_SIZE_BYTES",
    "LENGTH_BITS",
    "MAX_LENGTH_VALUE",
    "NONLINEAR_MAP_CODES",
    "NONLINEAR_MAP_FROM_CODE",
    "OccupancyComponent",
    "PsdFrame",
    "QuantizationState",
    "SERIALIZATION_FLAG_HAS_SCALE",
    "SERIALIZATION_MAGIC",
    "SERIALIZATION_VERSION",
    "StandardizationStats",
    "apply_nonlinear_map",
    "block_center_positions",
    "compute_block_layout",
    "compute_frame_metrics",
    "compute_standardization_stats",
    "db_to_linear_power",
    "decode_fwht_frame",
    "decimate_psd",
    "dequantize_symmetric_uniform",
    "deserialize_payload",
    "destandardize_values",
    "encode_fwht_frame",
    "estimate_noise_floor_db",
    "estimate_payload_bits",
    "evaluate_codec_dataset",
    "extract_occupied_components",
    "fwht_orthonormal",
    "linear_power_to_db",
    "load_psd_frames",
    "make_frequency_axis_hz",
    "materialize_sparse_coefficients",
    "next_power_of_two",
    "occupancy_mask",
    "parse_psd_values",
    "quantized_level_dtype",
    "quantize_float32_scalar",
    "quantize_symmetric_uniform",
    "reconstruct_fwht_frame",
    "select_representative_frame",
    "serialize_payload",
    "spectral_centroid_hz",
    "spectral_peak_frequency_hz",
    "standardize_values",
    "summarize_results",
    "total_occupied_bandwidth_hz",
    "upsample_psd",
    "validate_payload",
]
