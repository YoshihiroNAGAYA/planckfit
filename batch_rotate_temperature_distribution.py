import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("/Volumes/Nagaya_ssd/SPring-8-2026-Apr/YNFeHxEOS01")
DEFAULT_OUTPUT_ROOT = Path("/Users/nagayayoshihiro/work/plankfit/save/YNFeHxEOS/YNFeHxEOS01")
DEFAULT_UP_CALIBRATION = Path("/Volumes/Nagaya_ssd/laser calibration/2026A/Standard/Up/Up_std_OD0.spe")
DEFAULT_DOWN_CALIBRATION = Path("/Volumes/Nagaya_ssd/laser calibration/2026A/Standard/Down/Down_std_OD0.spe")
DEFAULT_LAMP_CSV = Path("/Volumes/Nagaya_ssd/laser calibration/2024B/OL245C.csv")

ROUND_ORDER = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run rotation correction and temperature-distribution export for all YNFeHxEOS01 rounds."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--up-calibration", type=Path, default=DEFAULT_UP_CALIBRATION)
    parser.add_argument("--down-calibration", type=Path, default=DEFAULT_DOWN_CALIBRATION)
    parser.add_argument("--lamp-csv", type=Path, default=DEFAULT_LAMP_CSV)
    parser.add_argument("--position-threshold", type=float, default=0.7)
    parser.add_argument(
        "--auto-angle-method",
        choices=["profile", "geometry"],
        default="profile",
    )
    parser.add_argument(
        "--rounds",
        nargs="*",
        default=list(ROUND_ORDER),
        help="Subset of rounds to process. Default: all 1st-8th.",
    )
    return parser.parse_args()


def collect_measurements(round_dir: Path) -> list[Path]:
    t_dir = round_dir / "T"
    if not t_dir.exists():
        return []
    return sorted(
        path for path in t_dir.glob("*.spe")
        if "_dist" not in path.stem.lower()
    )


def main():
    args = parse_args()
    script_path = Path(__file__).with_name("rotate_temperature_distribution.py")
    args.output_root.mkdir(parents=True, exist_ok=True)

    selected_rounds = [name for name in ROUND_ORDER if name in set(args.rounds)]
    if not selected_rounds:
        raise SystemExit("No valid rounds selected.")

    total_files = 0
    for round_name in selected_rounds:
        total_files += len(collect_measurements(args.input_root / round_name))

    print(f"input_root={args.input_root}")
    print(f"output_root={args.output_root}")
    print(f"auto_angle_method={args.auto_angle_method}")
    print(f"position_threshold={args.position_threshold}")
    print(f"total_files={total_files}")

    processed = 0
    for round_name in selected_rounds:
        round_dir = args.input_root / round_name
        measurements = collect_measurements(round_dir)
        output_dir = args.output_root / round_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{round_name}] files={len(measurements)} output_dir={output_dir}")
        if not measurements:
            continue

        for measurement in measurements:
            processed += 1
            print(f"  ({processed}/{total_files}) {measurement.name}")
            cmd = [
                sys.executable,
                str(script_path),
                "--measurement",
                str(measurement),
                "--up-calibration",
                str(args.up_calibration),
                "--down-calibration",
                str(args.down_calibration),
                "--lamp-csv",
                str(args.lamp_csv),
                "--output-dir",
                str(output_dir),
                "--position-threshold",
                str(args.position_threshold),
                "--auto-angle-method",
                args.auto_angle_method,
            ]
            subprocess.run(cmd, check=True)

    print("\nBatch processing completed.")


if __name__ == "__main__":
    main()
