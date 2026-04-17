import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.constants import c, h, k
from scipy.ndimage import rotate
from scipy.optimize import curve_fit


ROOT = Path("/Users/nagayayoshihiro/work")
PLANKFIT_DIR = ROOT / "plankfit"
SUPERIONIC_DIR = ROOT / "superionic"

if str(SUPERIONIC_DIR) not in sys.path:
    sys.path.insert(0, str(SUPERIONIC_DIR))

from modules.file_format.spe_wrapper import SpeWrapper


DEFAULT_MEASUREMENT = Path(
    "/Volumes/Nagaya_ssd/SPring-8-2026-Apr/YNFeHxEOS01/IHDAC/7th/T/YNFeHxEOS01_ 08.spe"
)
DEFAULT_UP_CALIBRATION = Path(
    "/Volumes/Nagaya_ssd/laser calibration/2026A/Standard/Up/Up_std_OD0.spe"
)
DEFAULT_DOWN_CALIBRATION = Path(
    "/Volumes/Nagaya_ssd/laser calibration/2026A/Standard/Down/Down_std_OD0.spe"
)
DEFAULT_LAMP_CSV = Path(
    "/Volumes/Nagaya_ssd/laser calibration/2024B/OL245C.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/nagayayoshihiro/work/plankfit/save/YNFeHxEOS/01/7th"
)

FIT_RANGE_NM = (600.0, 800.0)
COLUMN_MARGIN = 80
CENTER_WINDOW_HALF_HEIGHT = 36
DEFAULT_POSITION_THRESHOLD = 0.7
POSITION_PLOT_MARGIN = 4
DEFAULT_AUTO_ANGLE_METHOD = "profile"


@dataclass
class StreamTilt:
    name: str
    slope_px_per_col: float
    angle_deg: float
    peak_row: int
    total_weight: float
    columns_used: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate image rotation, rotate the spectrum image, and compare temperature distributions."
    )
    parser.add_argument("--measurement", type=Path, default=DEFAULT_MEASUREMENT)
    parser.add_argument("--up-calibration", type=Path, default=DEFAULT_UP_CALIBRATION)
    parser.add_argument("--down-calibration", type=Path, default=DEFAULT_DOWN_CALIBRATION)
    parser.add_argument("--lamp-csv", type=Path, default=DEFAULT_LAMP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manual-angle-deg",
        type=float,
        default=None,
        help="Skip auto-estimation and rotate by this angle.",
    )
    parser.add_argument(
        "--manual-up-angle-deg",
        type=float,
        default=None,
        help="Rotate only the upper stream by this angle.",
    )
    parser.add_argument(
        "--manual-down-angle-deg",
        type=float,
        default=None,
        help="Rotate only the lower stream by this angle.",
    )
    parser.add_argument(
        "--search-angle-deg",
        type=float,
        default=0.5,
        help="Auto-estimation search range in degrees.",
    )
    parser.add_argument(
        "--position-threshold",
        type=float,
        default=DEFAULT_POSITION_THRESHOLD,
        help="Relative threshold used to select temperature-fit rows in each stream.",
    )
    parser.add_argument(
        "--auto-angle-method",
        choices=["profile", "geometry"],
        default=DEFAULT_AUTO_ANGLE_METHOD,
        help="Automatic angle estimation method.",
    )
    return parser.parse_args()


def planck_function(wavelength_nm, temperature, scale):
    wavelength_m = wavelength_nm * 1e-9
    numerator = 2 * h * c**2
    denominator = wavelength_m**5
    exponent = (h * c) / (wavelength_m * k * temperature)
    return scale * numerator / denominator / (np.exp(exponent) - 1)


def fit_line_to_stream(sub_image: np.ndarray, row_offset: int, name: str) -> StreamTilt | None:
    row_sum = sub_image.sum(axis=1)
    if row_sum.size == 0 or np.all(~np.isfinite(row_sum)):
        return None

    peak_row = int(np.argmax(row_sum))
    y_index = np.arange(sub_image.shape[0], dtype=float)
    columns = []
    centers = []
    weights = []

    for col in range(COLUMN_MARGIN, sub_image.shape[1] - COLUMN_MARGIN):
        profile = sub_image[:, col].astype(float)
        baseline = np.percentile(profile, 25)
        profile = profile - baseline
        lo = max(0, peak_row - CENTER_WINDOW_HALF_HEIGHT)
        hi = min(sub_image.shape[0], peak_row + CENTER_WINDOW_HALF_HEIGHT + 1)
        windowed = np.zeros_like(profile)
        windowed[lo:hi] = profile[lo:hi]
        windowed[windowed < 0] = 0
        signal = float(windowed.sum())
        if signal <= 0:
            continue
        columns.append(col)
        centers.append(float(np.dot(y_index, windowed) / signal) + row_offset)
        weights.append(signal)

    if len(columns) < 20:
        return None

    fit = np.polyfit(columns, centers, 1, w=np.sqrt(np.asarray(weights)))
    slope = float(fit[0])
    return StreamTilt(
        name=name,
        slope_px_per_col=slope,
        angle_deg=float(np.degrees(np.arctan(slope))),
        peak_row=row_offset + peak_row,
        total_weight=float(np.sum(weights)),
        columns_used=len(columns),
    )


def evaluate_rotation_score(image: np.ndarray, angle_deg: float) -> tuple[float, list[StreamTilt]]:
    rotated = rotate_image_by_stream(image, angle_deg, angle_deg)
    mid = rotated.shape[0] // 2
    stream_metrics = [
        fit_line_to_stream(rotated[:mid, :], 0, "up"),
        fit_line_to_stream(rotated[mid:, :], mid, "down"),
    ]
    stream_metrics = [item for item in stream_metrics if item is not None]
    if not stream_metrics:
        return float("inf"), []
    weighted_score = sum(
        abs(item.slope_px_per_col) * item.total_weight for item in stream_metrics
    ) / sum(item.total_weight for item in stream_metrics)
    return float(weighted_score), stream_metrics


def estimate_rotation_angle(image: np.ndarray, search_angle_deg: float) -> tuple[float, list[StreamTilt], float]:
    coarse_angles = np.linspace(-search_angle_deg, search_angle_deg, 201)
    best_angle = 0.0
    best_score = float("inf")
    best_metrics: list[StreamTilt] = []

    for angle in coarse_angles:
        score, metrics = evaluate_rotation_score(image, float(angle))
        if score < best_score:
            best_angle = float(angle)
            best_score = score
            best_metrics = metrics

    fine_angles = np.linspace(best_angle - 0.05, best_angle + 0.05, 201)
    for angle in fine_angles:
        score, metrics = evaluate_rotation_score(image, float(angle))
        if score < best_score:
            best_angle = float(angle)
            best_score = score
            best_metrics = metrics

    return best_angle, best_metrics, best_score


def rotate_image_by_stream(image: np.ndarray, up_angle_deg: float, down_angle_deg: float) -> np.ndarray:
    mid = image.shape[0] // 2
    rotated_up = rotate(image[:mid, :], angle=up_angle_deg, reshape=False, order=1, mode="nearest")
    rotated_down = rotate(image[mid:, :], angle=down_angle_deg, reshape=False, order=1, mode="nearest")
    return np.vstack([rotated_up, rotated_down])


def get_stream_metrics_and_score(image: np.ndarray) -> tuple[list[StreamTilt], float]:
    mid = image.shape[0] // 2
    metrics = [
        fit_line_to_stream(image[:mid, :], 0, "up"),
        fit_line_to_stream(image[mid:, :], mid, "down"),
    ]
    metrics = [item for item in metrics if item is not None]
    if not metrics:
        return [], float("inf")
    score = sum(
        abs(item.slope_px_per_col) * item.total_weight for item in metrics
    ) / sum(item.total_weight for item in metrics)
    return metrics, float(score)


def load_reference_lamp(lamp_csv: Path) -> pd.DataFrame:
    lamp = pd.read_csv(lamp_csv, header=None, names=["wavelength", "intensity"])
    lamp = lamp.dropna()
    return lamp


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    spectrum = SpeWrapper(str(path))
    return spectrum.get_frame_data(frame=0), spectrum.get_wavelengths()[0]


def load_calibration_row(path: Path) -> np.ndarray:
    return SpeWrapper(str(path)).get_frame_data(frame=0).astype(float).reshape(-1)


def get_position_range(row_sum: np.ndarray, threshold: float, stream: str) -> np.ndarray:
    mid = row_sum.size // 2
    if stream == "up":
        extracted = row_sum[:mid]
        offset = 0
    elif stream == "down":
        extracted = row_sum[mid:]
        offset = mid
    else:
        raise ValueError(f"Unknown stream: {stream}")

    stream_max = float(np.max(extracted))
    stream_min = float(np.min(extracted))
    intensity_threshold = (stream_max - stream_min) * threshold + stream_min
    return np.where(extracted > intensity_threshold)[0] + offset


def get_position_range_for_subimage(sub_row_sum: np.ndarray, threshold: float) -> np.ndarray:
    stream_max = float(np.max(sub_row_sum))
    stream_min = float(np.min(sub_row_sum))
    intensity_threshold = (stream_max - stream_min) * threshold + stream_min
    return np.where(sub_row_sum > intensity_threshold)[0]


def fit_temperature_distribution(
    image: np.ndarray,
    wavelength_nm: np.ndarray,
    lamp_interp: np.ndarray,
    up_calibration: np.ndarray,
    down_calibration: np.ndarray,
    position_threshold: float,
) -> np.ndarray:
    temperatures = np.full(image.shape[0], np.nan, dtype=float)
    mid = image.shape[0] // 2
    fit_mask = (wavelength_nm >= FIT_RANGE_NM[0]) & (wavelength_nm <= FIT_RANGE_NM[1])
    row_sum = image.sum(axis=1)
    valid_rows = np.concatenate(
        [
            get_position_range(row_sum, position_threshold, "up"),
            get_position_range(row_sum, position_threshold, "down"),
        ]
    )
    valid_row_set = set(int(row) for row in valid_rows.tolist())

    for row in range(image.shape[0]):
        if row not in valid_row_set:
            continue
        raw = image[row].astype(float)
        if not np.isfinite(raw).all():
            continue

        calibration = up_calibration if row < mid else down_calibration
        corrected = lamp_interp * raw / calibration
        x = wavelength_nm[fit_mask]
        y = corrected[fit_mask]
        valid = np.isfinite(y) & (y > 0)
        if np.count_nonzero(valid) < 30:
            continue

        try:
            params, _ = curve_fit(
                planck_function,
                x[valid],
                y[valid],
                p0=[2000.0, 1e-18],
                bounds=([300.0, 0.0], [10000.0, np.inf]),
                maxfev=20000,
            )
        except Exception:
            continue

        temperatures[row] = float(params[0])

    return temperatures


def summarize_temperature_stats(temperatures: np.ndarray, stream_slice: slice) -> dict[str, float]:
    values = temperatures[stream_slice]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "std": np.nan}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def fit_temperature_profile_for_stream(
    sub_image: np.ndarray,
    wavelength_nm: np.ndarray,
    lamp_interp: np.ndarray,
    calibration: np.ndarray,
    position_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_sum = sub_image.sum(axis=1)
    positions = get_position_range_for_subimage(row_sum, position_threshold)
    temperatures = []
    fit_mask = (wavelength_nm >= FIT_RANGE_NM[0]) & (wavelength_nm <= FIT_RANGE_NM[1])

    for row in positions:
        raw = sub_image[row].astype(float)
        corrected = lamp_interp * raw / calibration
        x = wavelength_nm[fit_mask]
        y = corrected[fit_mask]
        valid = np.isfinite(y) & (y > 0)
        if np.count_nonzero(valid) < 30:
            temperatures.append(np.nan)
            continue
        try:
            params, _ = curve_fit(
                planck_function,
                x[valid],
                y[valid],
                p0=[2000.0, 1e-18],
                bounds=([300.0, 0.0], [10000.0, np.inf]),
                maxfev=20000,
            )
            temperatures.append(float(params[0]))
        except Exception:
            temperatures.append(np.nan)

    return positions, np.asarray(temperatures, dtype=float), row_sum


def compute_profile_similarity(
    positions: np.ndarray,
    temperatures: np.ndarray,
    row_sum: np.ndarray,
) -> float:
    if positions.size == 0 or temperatures.size == 0:
        return -np.inf
    intensity = row_sum[positions]
    valid = np.isfinite(temperatures) & np.isfinite(intensity) & (intensity > 0)
    if np.count_nonzero(valid) < 3:
        return -np.inf

    temp_profile = temperatures[valid]
    intensity_profile = np.log10(intensity[valid])
    temp_std = float(np.std(temp_profile))
    intensity_std = float(np.std(intensity_profile))
    if temp_std <= 0 or intensity_std <= 0:
        return -np.inf

    similarity = np.corrcoef(temp_profile, intensity_profile)[0, 1]
    return float(similarity)


def estimate_stream_angle_by_profile(
    sub_image: np.ndarray,
    wavelength_nm: np.ndarray,
    lamp_interp: np.ndarray,
    calibration: np.ndarray,
    position_threshold: float,
    search_angle_deg: float,
) -> tuple[float, float]:
    def score_angle(angle_deg: float) -> float:
        rotated_sub = rotate(sub_image, angle=angle_deg, reshape=False, order=1, mode="nearest")
        positions, temperatures, row_sum = fit_temperature_profile_for_stream(
            rotated_sub,
            wavelength_nm,
            lamp_interp,
            calibration,
            position_threshold,
        )
        return compute_profile_similarity(positions, temperatures, row_sum)

    coarse_angles = np.linspace(-search_angle_deg, search_angle_deg, 81)
    best_angle = 0.0
    best_score = -np.inf
    for angle in coarse_angles:
        score = score_angle(float(angle))
        if score > best_score:
            best_angle = float(angle)
            best_score = score

    fine_angles = np.linspace(best_angle - 0.05, best_angle + 0.05, 81)
    for angle in fine_angles:
        score = score_angle(float(angle))
        if score > best_score:
            best_angle = float(angle)
            best_score = score

    return best_angle, best_score


def save_rotation_preview(
    before: np.ndarray,
    after: np.ndarray,
    up_angle_deg: float,
    down_angle_deg: float,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    vmin = float(np.percentile(before, 5))
    vmax = float(np.percentile(before, 99.5))

    axes[0].imshow(before, aspect="auto", cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
    axes[0].set_title("Before rotation")
    axes[0].set_xlabel("Wavelength pixel")
    axes[0].set_ylabel("Position i")

    axes[1].imshow(after, aspect="auto", cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
    if abs(up_angle_deg - down_angle_deg) < 5e-7:
        after_title = f"After rotation ({up_angle_deg:.4f} deg)"
    else:
        after_title = f"After rotation (up={up_angle_deg:.4f}, down={down_angle_deg:.4f} deg)"
    axes[1].set_title(after_title)
    axes[1].set_xlabel("Wavelength pixel")
    axes[1].set_ylabel("Position i")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_temperature_comparison(
    before_t: np.ndarray,
    after_t: np.ndarray,
    before_intensity: np.ndarray,
    after_intensity: np.ndarray,
    up_profile_similarity: float | None,
    down_profile_similarity: float | None,
    output_path: Path,
):
    rows = np.arange(before_t.size)
    mid = before_t.size // 2
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), constrained_layout=True, sharex=False)
    stream_specs = [
        ("Upstream", slice(0, mid), up_profile_similarity),
        ("Downstream", slice(mid, before_t.size), down_profile_similarity),
    ]

    for ax, (title, stream_slice, similarity) in zip(axes, stream_specs):
        ax2 = ax.twinx()
        stream_rows = rows[stream_slice]
        stream_before_t = before_t[stream_slice]
        stream_after_t = after_t[stream_slice]
        stream_before_i = before_intensity[stream_slice]
        stream_after_i = after_intensity[stream_slice]
        before_stats = summarize_temperature_stats(before_t, stream_slice)
        after_stats = summarize_temperature_stats(after_t, stream_slice)

        plot_positions = np.unique(
            np.concatenate(
                [
                    np.flatnonzero(np.isfinite(stream_before_t)) + stream_rows[0],
                    np.flatnonzero(np.isfinite(stream_after_t)) + stream_rows[0],
                ]
            )
        )

        ax.plot(stream_rows, stream_before_t, color="#c44e52", lw=1.8, label="Before rotation")
        ax.plot(stream_rows, stream_after_t, color="#4c72b0", lw=1.8, label="After rotation")
        ax2.plot(stream_rows, stream_before_i, color="#dd8452", lw=1.2, ls="--", alpha=0.9, label="log Intensity before")
        ax2.plot(stream_rows, stream_after_i, color="#55a868", lw=1.2, ls=":", alpha=0.9, label="log Intensity after")

        ax.set_title(title)
        ax.set_ylabel("Temperature Ti (K)")
        ax2.set_ylabel("Integrated intensity (a.u.)")
        ax.set_ylim(bottom=1000)

        intensity_values = np.concatenate([stream_before_i, stream_after_i])
        positive_intensity = intensity_values[np.isfinite(intensity_values) & (intensity_values > 0)]
        if positive_intensity.size:
            ax2.set_yscale("log")
            ax2.set_ylim(positive_intensity.min() * 0.8, positive_intensity.max() * 1.2)

        if plot_positions.size:
            xmin = max(stream_rows[0], int(plot_positions.min()) - POSITION_PLOT_MARGIN)
            xmax = min(stream_rows[-1], int(plot_positions.max()) + POSITION_PLOT_MARGIN)
            ax.set_xlim(xmin, xmax)

        lines = ax.get_lines() + ax2.get_lines()
        lines = [line for line in lines if not line.get_label().startswith("_")]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, frameon=False, loc="upper right")
        ax.grid(alpha=0.2)

        similarity_text = "n/a" if similarity is None or not np.isfinite(similarity) else f"{similarity:.3f}"
        before_mean_text = "n/a" if not np.isfinite(before_stats["mean"]) else f"{before_stats['mean']:.1f}"
        before_std_text = "n/a" if not np.isfinite(before_stats["std"]) else f"{before_stats['std']:.1f}"
        after_mean_text = "n/a" if not np.isfinite(after_stats["mean"]) else f"{after_stats['mean']:.1f}"
        after_std_text = "n/a" if not np.isfinite(after_stats["std"]) else f"{after_stats['std']:.1f}"
        stats_text = (
            f"corr(log I, T) = {similarity_text}\n"
            f"before: mean={before_mean_text} K, std={before_std_text} K\n"
            f"after: mean={after_mean_text} K, std={after_std_text} K"
        )
        ax.text(
            0.02,
            0.05,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
        )

    axes[-1].set_xlabel("Position i")
    fig.suptitle("Temperature distribution before/after rotation")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_temperature_csv(before_t: np.ndarray, after_t: np.ndarray, output_path: Path):
    df = pd.DataFrame(
        {
            "position_i": np.arange(before_t.size),
            "before_temperature_K": before_t,
            "after_temperature_K": after_t,
        }
    )
    df.to_csv(output_path, index=False)


def save_summary_json(
    output_path: Path,
    measurement: Path,
    up_angle_deg: float,
    down_angle_deg: float,
    score: float,
    auto_angle_method: str,
    up_profile_similarity: float | None,
    down_profile_similarity: float | None,
    position_threshold: float,
    before_position_range: np.ndarray,
    after_position_range: np.ndarray,
    metrics_before: list[StreamTilt],
    metrics_after: list[StreamTilt],
):
    payload = {
        "measurement": str(measurement),
        "up_rotation_angle_deg": up_angle_deg,
        "down_rotation_angle_deg": down_angle_deg,
        "rotation_score": score,
        "auto_angle_method": auto_angle_method,
        "up_profile_similarity": up_profile_similarity,
        "down_profile_similarity": down_profile_similarity,
        "position_threshold": position_threshold,
        "before_position_range": before_position_range.tolist(),
        "after_position_range": after_position_range.tolist(),
        "before_stream_tilts": [
            {
                "name": item.name,
                "slope_px_per_col": item.slope_px_per_col,
                "angle_deg": item.angle_deg,
                "peak_row": item.peak_row,
                "total_weight": item.total_weight,
                "columns_used": item.columns_used,
            }
            for item in metrics_before
        ],
        "after_stream_tilts": [
            {
                "name": item.name,
                "slope_px_per_col": item.slope_px_per_col,
                "angle_deg": item.angle_deg,
                "peak_row": item.peak_row,
                "total_weight": item.total_weight,
                "columns_used": item.columns_used,
            }
            for item in metrics_after
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def format_angle_for_filename(angle_deg: float) -> str:
    sign = "p" if angle_deg >= 0 else "m"
    scaled = int(round(abs(angle_deg) * 1000))
    return f"{sign}{scaled:04d}mdeg"


def build_angle_tag(up_angle_deg: float, down_angle_deg: float) -> str:
    if abs(up_angle_deg - down_angle_deg) < 5e-7:
        return format_angle_for_filename(up_angle_deg)
    return f"up{format_angle_for_filename(up_angle_deg)}_down{format_angle_for_filename(down_angle_deg)}"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    before_image, wavelength_nm = load_image(args.measurement)
    up_calibration = load_calibration_row(args.up_calibration)
    down_calibration = load_calibration_row(args.down_calibration)
    lamp = load_reference_lamp(args.lamp_csv)
    lamp_interp = np.interp(wavelength_nm, lamp["wavelength"], lamp["intensity"])
    before_intensity = before_image.sum(axis=1)
    mid = before_image.shape[0] // 2

    _, metrics_before = evaluate_rotation_score(before_image, 0.0)
    up_profile_similarity = None
    down_profile_similarity = None
    if args.manual_up_angle_deg is not None or args.manual_down_angle_deg is not None:
        up_angle_deg = float(args.manual_up_angle_deg if args.manual_up_angle_deg is not None else 0.0)
        down_angle_deg = float(args.manual_down_angle_deg if args.manual_down_angle_deg is not None else 0.0)
        after_image = rotate_image_by_stream(before_image, up_angle_deg, down_angle_deg)
    elif args.manual_angle_deg is None:
        if args.auto_angle_method == "profile":
            up_angle_deg, up_profile_similarity = estimate_stream_angle_by_profile(
                before_image[:mid, :],
                wavelength_nm,
                lamp_interp,
                up_calibration,
                args.position_threshold,
                args.search_angle_deg,
            )
            down_angle_deg, down_profile_similarity = estimate_stream_angle_by_profile(
                before_image[mid:, :],
                wavelength_nm,
                lamp_interp,
                down_calibration,
                args.position_threshold,
                args.search_angle_deg,
            )
        else:
            angle_deg, _, _ = estimate_rotation_angle(before_image, args.search_angle_deg)
            up_angle_deg = angle_deg
            down_angle_deg = angle_deg
        after_image = rotate_image_by_stream(before_image, up_angle_deg, down_angle_deg)
    else:
        up_angle_deg = float(args.manual_angle_deg)
        down_angle_deg = float(args.manual_angle_deg)
        after_image = rotate_image_by_stream(before_image, up_angle_deg, down_angle_deg)

    metrics_after, after_score = get_stream_metrics_and_score(after_image)
    after_intensity = after_image.sum(axis=1)
    before_position_range = np.concatenate(
        [
            get_position_range(before_intensity, args.position_threshold, "up"),
            get_position_range(before_intensity, args.position_threshold, "down"),
        ]
    )
    after_position_range = np.concatenate(
        [
            get_position_range(after_intensity, args.position_threshold, "up"),
            get_position_range(after_intensity, args.position_threshold, "down"),
        ]
    )
    before_temperatures = fit_temperature_distribution(
        before_image,
        wavelength_nm,
        lamp_interp,
        up_calibration,
        down_calibration,
        args.position_threshold,
    )
    after_temperatures = fit_temperature_distribution(
        after_image,
        wavelength_nm,
        lamp_interp,
        up_calibration,
        down_calibration,
        args.position_threshold,
    )

    stem = args.measurement.stem.replace(" ", "_")
    angle_tag = build_angle_tag(up_angle_deg, down_angle_deg)
    preview_path = args.output_dir / f"{stem}_{angle_tag}_rotation_preview.png"
    comparison_path = args.output_dir / f"{stem}_{angle_tag}_temperature_distribution_before_after.png"
    csv_path = args.output_dir / f"{stem}_{angle_tag}_temperature_distribution.csv"
    summary_path = args.output_dir / f"{stem}_{angle_tag}_rotation_summary.json"

    save_rotation_preview(before_image, after_image, up_angle_deg, down_angle_deg, preview_path)
    save_temperature_comparison(
        before_temperatures,
        after_temperatures,
        before_intensity,
        after_intensity,
        up_profile_similarity,
        down_profile_similarity,
        comparison_path,
    )
    save_temperature_csv(before_temperatures, after_temperatures, csv_path)
    save_summary_json(
        summary_path,
        args.measurement,
        up_angle_deg,
        down_angle_deg,
        after_score,
        args.auto_angle_method,
        up_profile_similarity,
        down_profile_similarity,
        args.position_threshold,
        before_position_range,
        after_position_range,
        metrics_before,
        metrics_after,
    )

    print(f"up_rotation_angle_deg={up_angle_deg:.6f}")
    print(f"down_rotation_angle_deg={down_angle_deg:.6f}")
    print(f"rotation_score={after_score:.8f}")
    if up_profile_similarity is not None:
        print(f"up_profile_similarity={up_profile_similarity:.6f}")
    if down_profile_similarity is not None:
        print(f"down_profile_similarity={down_profile_similarity:.6f}")
    print(f"rotation_preview={preview_path}")
    print(f"temperature_comparison={comparison_path}")
    print(f"temperature_csv={csv_path}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
