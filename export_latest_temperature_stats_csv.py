import json
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROUND_ORDER = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]


def parse_args():
    parser = argparse.ArgumentParser(description="Export latest upstream/downstream temperature statistics as CSV.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    return parser.parse_args()


def collect_latest_summaries(measurement_root: Path, save_root: Path) -> list[tuple[str, Path, str]]:
    latest_by_measurement: dict[str, Path] = {}
    latest_mtime: dict[str, float] = {}

    for round_name in ROUND_ORDER:
        round_dir = save_root / round_name
        expected_prefix = str(measurement_root / round_name / "T") + "/"
        if not round_dir.exists():
            continue
        for summary_path in round_dir.glob("*rotation_summary.json"):
            payload = json.loads(summary_path.read_text())
            measurement = payload["measurement"]
            if not measurement.startswith(expected_prefix):
                continue
            mtime = summary_path.stat().st_mtime
            if measurement not in latest_mtime or mtime > latest_mtime[measurement]:
                latest_mtime[measurement] = mtime
                latest_by_measurement[measurement] = summary_path

    rows = []
    for measurement, summary_path in sorted(latest_by_measurement.items()):
        rows.append((measurement, summary_path, summary_path.parent.name))
    return rows


def extract_measurement_name(measurement_path: str) -> str:
    match = re.search(r"_(\s*\d+)\.spe$", measurement_path, re.IGNORECASE)
    if not match:
        return Path(measurement_path).stem
    return str(int(match.group(1)))


def compute_stream_stats(temperature_csv_path: Path) -> tuple[float, float, float, float]:
    df = pd.read_csv(temperature_csv_path)
    mid = len(df) // 2

    up_values = df.loc[df["position_i"] < mid, "after_temperature_K"].to_numpy(dtype=float)
    down_values = df.loc[df["position_i"] >= mid, "after_temperature_K"].to_numpy(dtype=float)

    up_values = up_values[np.isfinite(up_values)]
    down_values = down_values[np.isfinite(down_values)]

    up_mean = float(np.mean(up_values)) if up_values.size else np.nan
    up_std = float(np.std(up_values)) if up_values.size else np.nan
    down_mean = float(np.mean(down_values)) if down_values.size else np.nan
    down_std = float(np.std(down_values)) if down_values.size else np.nan

    return up_mean, up_std, down_mean, down_std


def main():
    args = parse_args()
    output_csv = args.save_root / "latest_temperature_stats.csv"
    records = []

    for measurement, summary_path, round_name in collect_latest_summaries(args.measurement_root, args.save_root):
        stem = summary_path.name.replace("_rotation_summary.json", "")
        temperature_csv_path = summary_path.with_name(f"{stem}_temperature_distribution.csv")
        if not temperature_csv_path.exists():
            continue

        up_mean, up_std, down_mean, down_std = compute_stream_stats(temperature_csv_path)
        records.append(
            {
                "run_name": args.run_name,
                "experiment_name": round_name,
                "measurement_name": extract_measurement_name(measurement),
                "up_mean_temperature_K": up_mean,
                "up_std_temperature_K": up_std,
                "down_mean_temperature_K": down_mean,
                "down_std_temperature_K": down_std,
                "measurement_path": measurement,
                "summary_path": str(summary_path),
                "temperature_csv_path": str(temperature_csv_path),
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df["measurement_name_num"] = pd.to_numeric(df["measurement_name"], errors="coerce")
        df = df.sort_values(["experiment_name", "measurement_name_num", "measurement_name"]).drop(columns=["measurement_name_num"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"records={len(df)}")
    print(f"csv={output_csv}")


if __name__ == "__main__":
    main()
