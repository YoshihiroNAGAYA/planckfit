import json
import textwrap
import uuid
from pathlib import Path


ROOT = Path("/Users/nagayayoshihiro/work/plankfit")
NOTEBOOK_PATH = ROOT / "dynamic_speed_temperature_comparison.ipynb"


def lines(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n")
    return [f"{line}\n" for line in normalized.splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown_cell(
        """
        # Kinetix Dynamic / speed 温度比較

        `/Users/nagayayoshihiro/work/superionic/notebook/temporal_temp_analysis.ipynb` の流れをベースに、
        `BL10XU_Kinetix_Lamp_Calibration_20260401` の Kinetix 校正データから
        Dynamic と speed の温度を比較するためのノートブックです。

        このノートブックでは以下を行います。
        - 校正ランプと参照スペクトルを読み込む
        - 上流 / 下流それぞれで Dynamic と speed の代表位置を自動抽出する
        - 600-800 nm の範囲でプランクフィットして温度を求める
        - Dynamic と speed の温度を表と図で比較する
        """
    ),
    code_cell(
        """
        import sys
        import warnings
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import seaborn as sns
        from matplotlib import pyplot as plt
        from scipy.constants import h, c, k
        from scipy.optimize import curve_fit

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sns.set_theme(style="ticks")

        ROOT = Path("/Users/nagayayoshihiro/work")
        SUPERIONIC_DIR = ROOT / "superionic"
        DATA_DIR = SUPERIONIC_DIR / "data" / "BL10XU_Kinetix_Lamp_Calibration_20260401"

        if str(SUPERIONIC_DIR) not in sys.path:
            sys.path.insert(0, str(SUPERIONIC_DIR))

        from modules.file_format.spe_wrapper import SpeWrapper
        from modules.data_model.spectrum_data import SpectrumData

        print(DATA_DIR)
        """
    ),
    code_cell(
        """
        expected_centers = {
            "up": 190,
            "dw": 809,
        }

        measurement_files = {
            ("up", "Dynamic"): DATA_DIR / "Kinetix_Up_3.914A_OD0_Dynamic_50msec_1frame_20260401-134425_flipped.spe",
            ("up", "speed"): DATA_DIR / "Kinetix_Up_3.914A_OD0_speed_950usec_1frame_20260401-134658_flipped.spe",
            ("dw", "Dynamic"): DATA_DIR / "Kinetix_DS_3.914A_OD0_Dynamic_50msec_20260331-170443_flipped.spe",
            ("dw", "speed"): DATA_DIR / "Kinetix_DS_3.914A_OD0_speed_950usec_20260331-170224_flipped.spe",
        }

        calibration_files = {
            "up": DATA_DIR / "Kinetix_up_std_OD0.spe",
            "dw": DATA_DIR / "Kinetix_ds_std_OD0.spe",
        }

        reference_csv = DATA_DIR / "OL245C.csv"
        """
    ),
    code_cell(
        """
        ref_spectrum = pd.read_csv(
            reference_csv,
            header=None,
            names=["wavelength", "intensity"],
        )

        calibration_spe = {stream: SpeWrapper(str(path)) for stream, path in calibration_files.items()}
        wavelength_arr = calibration_spe["up"].get_wavelengths()[0]
        ref_intensity_interpolated = np.interp(
            wavelength_arr,
            ref_spectrum["wavelength"],
            ref_spectrum["intensity"],
        )
        calibration_spectrum = {
            stream: spe.get_all_data_arr()[0][0]
            for stream, spe in calibration_spe.items()
        }

        plt.figure(figsize=(8, 3))
        plt.plot(ref_spectrum["wavelength"], ref_spectrum["intensity"], color="black")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Radiance")
        plt.title("Reference lamp spectrum")
        plt.show()
        """
    ),
    code_cell(
        """
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
            local_peak = np.argmax(intensity_sum[lo:hi])
            peak_position = lo + int(local_peak)
            return peak_position, intensity_sum


        def fit_temperature_from_raw_spectrum(raw_spectrum, calibration_reference):
            corrected = ref_intensity_interpolated * raw_spectrum / calibration_reference
            mask = (wavelength_arr >= FIT_RANGE[0]) & (wavelength_arr <= FIT_RANGE[1])

            x = wavelength_arr[mask]
            y = corrected[mask]
            valid = np.isfinite(y) & (y > 0)
            x = x[valid]
            y = y[valid]

            if len(x) < 10:
                return {
                    "temperature": np.nan,
                    "temperature_error": np.nan,
                    "wavelength_fit": x,
                    "intensity_fit": y,
                    "fitted_curve": np.full_like(x, np.nan, dtype=float),
                }

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


        def summarize_measurement(stream, mode, path):
            spectrum_data = SpectrumData(str(path))
            image = spectrum_data.get_frame_data(0)
            peak_position, intensity_sum = find_hotspot_position(image, expected_centers[stream])

            pixel_results = []
            lo = max(0, peak_position - AVERAGE_HALF_WIDTH)
            hi = min(image.shape[0], peak_position + AVERAGE_HALF_WIDTH + 1)

            for position in range(lo, hi):
                fit_result = fit_temperature_from_raw_spectrum(
                    raw_spectrum=image[position],
                    calibration_reference=calibration_spectrum[stream],
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
                "peak_count": float(image[peak_position].max()),
                "mean_temperature": float(np.mean(temperature_values)),
                "std_temperature": float(np.std(temperature_values)),
                "pixel_results": pixel_results,
                "representative_fit": representative,
                "image": image,
                "intensity_sum": intensity_sum,
            }
        """
    ),
    code_cell(
        """
        results = []
        for (stream, mode), path in measurement_files.items():
            summary = summarize_measurement(stream, mode, path)
            results.append(summary)

        summary_df = pd.DataFrame(
            [
                {
                    "stream": item["stream"],
                    "mode": item["mode"],
                    "peak_position": item["peak_position"],
                    "peak_count": item["peak_count"],
                    "mean_temperature_K": item["mean_temperature"],
                    "std_temperature_K": item["std_temperature"],
                }
                for item in results
            ]
        ).sort_values(["stream", "mode"]).reset_index(drop=True)

        summary_df
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

        for ax, item in zip(axes.flat, results):
            rep = item["representative_fit"]
            ax.plot(rep["wavelength_fit"], rep["intensity_fit"], label="corrected spectrum", lw=2)
            ax.plot(rep["wavelength_fit"], rep["fitted_curve"], label="Planck fit", lw=2, ls="--")
            ax.set_title(
                f"{item['stream']} / {item['mode']} | "
                f"T = {item['mean_temperature']:.1f} K"
            )
            ax.set_ylabel("Intensity (a.u.)")
            ax.grid(False)
            ax.legend()

        for ax in axes[-1]:
            ax.set_xlabel("Wavelength (nm)")

        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(
            data=summary_df,
            x="stream",
            y="mean_temperature_K",
            hue="mode",
            palette={"Dynamic": "#d55e00", "speed": "#0072b2"},
        )

        for patch, (_, row) in zip(ax.patches, summary_df.iterrows()):
            x = patch.get_x() + patch.get_width() / 2
            y = row["mean_temperature_K"]
            err = row["std_temperature_K"]
            ax.errorbar(x=x, y=y, yerr=err, color="black", capsize=4, lw=1.2)

        ax.set_xlabel("Stream")
        ax.set_ylabel("Temperature (K)")
        ax.set_title("Dynamic vs speed temperature comparison")
        ax.grid(False)
        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        comparison_df = (
            summary_df.pivot(index="stream", columns="mode", values="mean_temperature_K")
            .assign(delta_speed_minus_dynamic=lambda df: df["speed"] - df["Dynamic"])
        )

        comparison_df
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "analysisenv",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


ROOT.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2))
print(NOTEBOOK_PATH)
