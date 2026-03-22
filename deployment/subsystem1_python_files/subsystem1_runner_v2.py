from pathlib import Path
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent


def run_file(filename: str) -> None:
    path = BASE_DIR / filename

    print(f"\n▶ Running: {filename}")

    try:
        subprocess.run([sys.executable, str(path)], check=True)
        print(f"Completed: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error in: {filename}")
        raise e


if __name__ == "__main__":
    print("\nStarting Subsystem 1 Pipeline...\n")

    run_file("rolling_window_builder.py")   # Step 1: build features
    run_file("forecast_model.py")           # Step 2: model prediction
    run_file("allocation_layer.py")         # Step 3: allocation logic

    print("\nSubsystem 1 pipeline finished successfully.\n")