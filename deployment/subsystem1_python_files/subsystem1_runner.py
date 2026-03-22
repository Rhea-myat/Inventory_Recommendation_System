from pathlib import Path
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent


def run_file(filename: str) -> None:
    path = BASE_DIR / filename
    subprocess.run([sys.executable, str(path)], check=True)


if __name__ == "__main__":
    run_file("rolling_window_builder.py")
    run_file("forecast_model.py")
    run_file("allocation_layer.py")
    print("Subsystem 1 pipeline finished.")