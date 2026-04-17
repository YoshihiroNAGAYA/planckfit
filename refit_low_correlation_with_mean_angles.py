import json
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


UP_CALIBRATION = Path("/Volumes/Nagaya_ssd/laser calibration/2026A/Standard/Up/Up_std_OD0.spe")
DOWN_CALIBRATION = Path("/Volumes/Nagaya_ssd/laser calibration/2026A/Standard/Down/Down_std_OD0.spe")
LAMP_CSV = Path("/Volumes/Nagaya_ssd/laser calibration/2024B/OL245C.csv")

GOOD_CORRELATION_THRESHOLD = 0.8
LOW_CORRELATION_THRESHOLD = 0.6
POSITION_THRESHOLD = 0.7
ROUND_ORDER = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]


def parse_args():
    parser = argparse.ArgumentParser(description="Refit low-correlation measurements with mean high-correlation angles.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--position-threshold", type=float, default=POSITION_THRESHOLD)
    return parser.parse_args()


def load_latest_summaries(measurement_root: Path, save_root: Path) -> pd.DataFrame:
    latest_by_measurement: dict[str, Path] = {}
    latest_mtime: dict[str, float] = {}

    for round_name in ROUND_ORDER:
        round_dir = save_root / round_name
        expected_prefix = str(measurement_root / round_name / "T") + "/"
        if not round_dir.exists():
            continue
        for path in round_dir.glob("*rotation_summary.json"):
            payload = json.loads(path.read_text())
            measurement = payload["measurement"]
            if not measurement.startswith(expected_prefix):
                continue
            mtime = path.stat().st_mtime
            if measurement not in latest_mtime or mtime > latest_mtime[measurement]:
                latest_mtime[measurement] = mtime
                latest_by_measurement[measurement] = path

    records = []
    for measurement, path in sorted(latest_by_measurement.items()):
        payload = json.loads(path.read_text())
        records.append(
            {
                "round": path.parent.name,
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


def main():
    args = parse_args()
    manifest_csv = args.save_root / "low_correlation_refit_manifest.csv"
    df = load_latest_summaries(args.measurement_root, args.save_root)
    if df.empty:
        raise SystemExit("No summary data found.")

    up_mean_angle = float(df.loc[df["up_profile_similarity"] >= GOOD_CORRELATION_THRESHOLD, "up_rotation_angle_deg"].mean())
    down_mean_angle = float(df.loc[df["down_profile_similarity"] >= GOOD_CORRELATION_THRESHOLD, "down_rotation_angle_deg"].mean())

    if pd.isna(up_mean_angle) or pd.isna(down_mean_angle):
        raise SystemExit("Could not compute mean angles from high-correlation results.")

    rotate_script = Path(__file__).with_name("rotate_temperature_distribution.py")
    manifest_records = []

    print(f"up_mean_angle_deg={up_mean_angle:.6f}")
    print(f"down_mean_angle_deg={down_mean_angle:.6f}")

    for row in df.itertuples(index=False):
        use_up_fallback = pd.notna(row.up_profile_similarity) and row.up_profile_similarity < LOW_CORRELATION_THRESHOLD
        use_down_fallback = pd.notna(row.down_profile_similarity) and row.down_profile_similarity < LOW_CORRELATION_THRESHOLD
        if not use_up_fallback and not use_down_fallback:
            continue

        output_dir = args.save_root / row.round
        manual_up_angle = up_mean_angle if use_up_fallback else float(row.up_rotation_angle_deg)
        manual_down_angle = down_mean_angle if use_down_fallback else float(row.down_rotation_angle_deg)

        print(
            f"refit {row.measurement_name} "
            f"(up_corr={row.up_profile_similarity:.3f}, down_corr={row.down_profile_similarity:.3f}) "
            f"-> up={manual_up_angle:.6f}, down={manual_down_angle:.6f}"
        )

        cmd = [
            sys.executable,
            str(rotate_script),
            "--measurement",
            row.measurement,
            "--up-calibration",
            str(UP_CALIBRATION),
            "--down-calibration",
            str(DOWN_CALIBRATION),
            "--lamp-csv",
            str(LAMP_CSV),
            "--output-dir",
            str(output_dir),
            "--position-threshold",
            str(args.position_threshold),
            "--manual-up-angle-deg",
            str(manual_up_angle),
            "--manual-down-angle-deg",
            str(manual_down_angle),
        ]
        subprocess.run(cmd, check=True)

        manifest_records.append(
            {
                "round": row.round,
                "measurement": row.measurement,
                "measurement_name": row.measurement_name,
                "original_up_angle_deg": row.up_rotation_angle_deg,
                "original_down_angle_deg": row.down_rotation_angle_deg,
                "original_up_profile_similarity": row.up_profile_similarity,
                "original_down_profile_similarity": row.down_profile_similarity,
                "used_up_fallback": use_up_fallback,
                "used_down_fallback": use_down_fallback,
                "fallback_up_angle_deg": manual_up_angle,
                "fallback_down_angle_deg": manual_down_angle,
                "global_good_up_mean_angle_deg": up_mean_angle,
                "global_good_down_mean_angle_deg": down_mean_angle,
            }
        )

    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"refit_count={len(manifest_df)}")
    print(f"manifest={manifest_csv}")


if __name__ == "__main__":
    main()
