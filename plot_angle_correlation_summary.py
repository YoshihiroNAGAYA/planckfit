import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt


SAVE_ROOT = Path("/Users/nagayayoshihiro/work/plankfit/save/YNFeHxEOS/YNFeHxEOS01")
ROUND_ORDER = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]
OUTPUT_PNG = SAVE_ROOT / "angle_vs_correlation_1st_7th.png"
OUTPUT_CSV = SAVE_ROOT / "angle_vs_correlation_1st_7th.csv"

ROUND_COLORS = {
    "1st": "#4c78a8",
    "2nd": "#f58518",
    "3rd": "#54a24b",
    "4th": "#e45756",
    "5th": "#72b7b2",
    "6th": "#b279a2",
    "7th": "#ff9da6",
}


def collect_latest_summaries() -> pd.DataFrame:
    latest_by_measurement: dict[str, Path] = {}
    latest_mtime: dict[str, float] = {}

    for round_name in ROUND_ORDER:
        round_dir = SAVE_ROOT / round_name
        if not round_dir.exists():
            continue
        for path in round_dir.glob("*rotation_summary.json"):
            payload = json.loads(path.read_text())
            measurement = payload["measurement"]
            expected_prefix = str(Path("/Volumes/Nagaya_ssd/SPring-8-2026-Apr/YNFeHxEOS01") / round_name / "T") + "/"
            if not measurement.startswith(expected_prefix):
                continue
            mtime = path.stat().st_mtime
            if measurement not in latest_mtime or mtime > latest_mtime[measurement]:
                latest_mtime[measurement] = mtime
                latest_by_measurement[measurement] = path

    records = []
    for measurement, path in sorted(latest_by_measurement.items()):
        payload = json.loads(path.read_text())
        round_name = path.parent.name
        records.append(
            {
                "round": round_name,
                "measurement": measurement,
                "measurement_name": Path(measurement).name,
                "summary_path": str(path),
                "up_rotation_angle_deg": payload.get("up_rotation_angle_deg"),
                "down_rotation_angle_deg": payload.get("down_rotation_angle_deg"),
                "up_profile_similarity": payload.get("up_profile_similarity"),
                "down_profile_similarity": payload.get("down_profile_similarity"),
            }
        )

    return pd.DataFrame(records)


def plot_dataframe(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), constrained_layout=True, sharex=False)
    stream_specs = [
        ("Upstream", "up_rotation_angle_deg", "up_profile_similarity"),
        ("Downstream", "down_rotation_angle_deg", "down_profile_similarity"),
    ]

    for ax, (title, angle_col, corr_col) in zip(axes, stream_specs):
        for round_name in ROUND_ORDER:
            subset = df[df["round"] == round_name]
            if subset.empty:
                continue
            ax.scatter(
                subset[angle_col],
                subset[corr_col],
                s=55,
                alpha=0.9,
                color=ROUND_COLORS[round_name],
                label=round_name,
                edgecolors="none",
            )

        ax.set_title(title)
        ax.set_xlabel("Rotation angle (deg)")
        ax.set_ylabel("Correlation coefficient")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=4, loc="lower center")

    fig.suptitle("Rotation angle vs correlation coefficient (1st-7th)")
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    df = collect_latest_summaries()
    if df.empty:
        raise SystemExit("No summary JSON files found.")

    df = df.sort_values(["round", "measurement_name"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    plot_dataframe(df)

    print(f"records={len(df)}")
    print(f"png={OUTPUT_PNG}")
    print(f"csv={OUTPUT_CSV}")


if __name__ == "__main__":
    main()
