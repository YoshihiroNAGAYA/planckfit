import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.constants import c, h, k
from scipy.optimize import curve_fit


ROOT = Path("/Users/nagayayoshihiro/work")
PLANKFIT_DIR = ROOT / "plankfit"
SUPERIONIC_DIR = ROOT / "superionic"
DATA_DIR = SUPERIONIC_DIR / "data" / "BL10XU_Kinetix_Lamp_Calibration_20260401"
OUTPUT_DIR = PLANKFIT_DIR / "pdf"

if str(SUPERIONIC_DIR) not in sys.path:
    sys.path.insert(0, str(SUPERIONIC_DIR))

from modules.file_format.spe_wrapper import SpeWrapper
from modules.data_model.spectrum_data import SpectrumData


EXPECTED_CENTERS = {
    "up": 190,
    "dw": 809,
}

MEASUREMENT_FILES = {
    ("up", "Dynamic"): DATA_DIR / "Kinetix_Up_3.914A_OD0_Dynamic_50msec_1frame_20260401-134425_flipped.spe",
    ("up", "speed"): DATA_DIR / "Kinetix_Up_3.914A_OD0_speed_950usec_1frame_20260401-134658_flipped.spe",
    ("dw", "Dynamic"): DATA_DIR / "Kinetix_DS_3.914A_OD0_Dynamic_50msec_20260331-170443_flipped.spe",
    ("dw", "speed"): DATA_DIR / "Kinetix_DS_3.914A_OD0_speed_950usec_20260331-170224_flipped.spe",
}

CALIBRATION_FILES = {
    "up": DATA_DIR / "Kinetix_up_std_OD0.spe",
    "dw": DATA_DIR / "Kinetix_ds_std_OD0.spe",
}

REFERENCE_CSV = DATA_DIR / "OL245C.csv"
FIT_RANGE = (600, 800)
SEARCH_HALF_WIDTH = 40
AVERAGE_HALF_WIDTH = 5


def planck_function(wavelength_nm, temperature, scale):
    wavelength_m = wavelength_nm * 1e-9
    numerator = 2 * h * c**2
    denominator = wavelength_m**5
    exponent = (h * c) / (wavelength_m * k * temperature)
    return scale * numerator / denominator / (np.exp(exponent) - 1)


def find_hotspot_position(image, expected_center, search_half_width=SEARCH_HALF_WIDTH):
    intensity_sum = image.sum(axis=1)
    lo = max(0, expected_center - search_half_width)
    hi = min(image.shape[0], expected_center + search_half_width + 1)
    peak_position = lo + int(np.argmax(intensity_sum[lo:hi]))
    return peak_position, intensity_sum


def fit_temperature_from_raw_spectrum(raw_spectrum, calibration_reference, wavelength_arr, ref_interp):
    corrected = ref_interp * raw_spectrum / calibration_reference
    mask = (wavelength_arr >= FIT_RANGE[0]) & (wavelength_arr <= FIT_RANGE[1])

    x = wavelength_arr[mask]
    y = corrected[mask]
    valid = np.isfinite(y) & (y > 0)
    x = x[valid]
    y = y[valid]

    params, covariance = curve_fit(
        planck_function,
        x,
        y,
        p0=[2000, 1e-18],
        bounds=([300, 0], [10000, np.inf]),
        maxfev=20000,
    )

    temperature, scale = params
    temperature_error = float(np.sqrt(np.diag(covariance))[0]) if covariance is not None else np.nan

    return {
        "temperature": float(temperature),
        "temperature_error": temperature_error,
        "wavelength_fit": x,
        "intensity_fit": y,
        "fitted_curve": planck_function(x, temperature, scale),
    }


def summarize_measurement(stream, mode, path, calibration_reference, wavelength_arr, ref_interp):
    spectrum_data = SpectrumData(str(path))
    image = spectrum_data.get_frame_data(0)
    peak_position, intensity_sum = find_hotspot_position(image, EXPECTED_CENTERS[stream])

    pixel_results = []
    lo = max(0, peak_position - AVERAGE_HALF_WIDTH)
    hi = min(image.shape[0], peak_position + AVERAGE_HALF_WIDTH + 1)

    for position in range(lo, hi):
        fit_result = fit_temperature_from_raw_spectrum(
            raw_spectrum=image[position],
            calibration_reference=calibration_reference,
            wavelength_arr=wavelength_arr,
            ref_interp=ref_interp,
        )
        fit_result["position"] = position
        fit_result["peak_count"] = float(image[position].max())
        fit_result["sum_count"] = float(intensity_sum[position])
        pixel_results.append(fit_result)

    temperature_values = [item["temperature"] for item in pixel_results if np.isfinite(item["temperature"])]
    representative = min(pixel_results, key=lambda item: abs(item["position"] - peak_position))

    return {
        "stream": stream,
        "mode": mode,
        "path": str(path),
        "peak_position": peak_position,
        "mean_temperature": float(np.mean(temperature_values)),
        "std_temperature": float(np.std(temperature_values)),
        "representative_fit": representative,
        "intensity_sum": intensity_sum,
    }


def set_adaptive_ylim(ax, rep):
    y_values = np.concatenate([rep["intensity_fit"], rep["fitted_curve"]])
    y_values = y_values[np.isfinite(y_values)]

    if y_values.size == 0:
        return

    y_min = float(y_values.min())
    y_max = float(y_values.max())
    y_span = y_max - y_min

    if y_span <= 0:
        margin = max(abs(y_max) * 0.1, 1.0)
    else:
        margin = y_span * 0.08

    lower = max(0.0, y_min - margin)
    upper = y_max + margin
    ax.set_ylim(lower, upper)


def export_stream_pdf(stream, stream_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)

    for ax, item in zip(axes, stream_results):
        rep = item["representative_fit"]
        ax.plot(rep["wavelength_fit"], rep["intensity_fit"], color="#1f77b4", lw=2, label="Corrected spectrum")
        ax.plot(rep["wavelength_fit"], rep["fitted_curve"], color="#d62728", lw=2, ls="--", label="Planck fit")
        set_adaptive_ylim(ax, rep)
        ax.set_title(
            f"{stream.upper()} / {item['mode']}\n"
            f"pixel={rep['position']}, T={item['mean_temperature']:.1f}±{item['std_temperature']:.1f} K"
        )
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Corrected intensity (a.u.)")
        ax.grid(False)
        ax.legend(frameon=False)

    fig.suptitle(f"{stream.upper()} stream: Dynamic vs speed")
    fig.tight_layout()

    out = OUTPUT_DIR / f"{stream}_dynamic_speed_spectrum_fit.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_spectrum = pd.read_csv(
        REFERENCE_CSV,
        header=None,
        names=["wavelength", "intensity"],
    )

    calibration_spe = {stream: SpeWrapper(str(path)) for stream, path in CALIBRATION_FILES.items()}
    wavelength_arr = calibration_spe["up"].get_wavelengths()[0]
    ref_interp = np.interp(wavelength_arr, ref_spectrum["wavelength"], ref_spectrum["intensity"])
    calibration_spectrum = {
        stream: spe.get_all_data_arr()[0][0]
        for stream, spe in calibration_spe.items()
    }

    results = []
    for (stream, mode), path in MEASUREMENT_FILES.items():
        results.append(
            summarize_measurement(
                stream=stream,
                mode=mode,
                path=path,
                calibration_reference=calibration_spectrum[stream],
                wavelength_arr=wavelength_arr,
                ref_interp=ref_interp,
            )
        )

    ordered_results = sorted(results, key=lambda item: (item["stream"], item["mode"]))
    summary_df = pd.DataFrame(
        [
            {
                "stream": item["stream"],
                "mode": item["mode"],
                "peak_position": item["peak_position"],
                "fit_position": item["representative_fit"]["position"],
                "mean_temperature_K": item["mean_temperature"],
                "std_temperature_K": item["std_temperature"],
            }
            for item in ordered_results
        ]
    )
    summary_df.to_csv(OUTPUT_DIR / "dynamic_speed_temperature_summary.csv", index=False)

    exported = []
    for stream in ("up", "dw"):
        stream_results = [item for item in ordered_results if item["stream"] == stream]
        exported.append(export_stream_pdf(stream, stream_results))

    print(summary_df.to_string(index=False))
    print("\nPDF files:")
    for path in exported:
        print(path)


if __name__ == "__main__":
    main()
