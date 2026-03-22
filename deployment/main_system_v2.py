from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent


def run_step(name: str, path: Path):
    print(f"\nRunning: {name}")

    try:
        subprocess.run([sys.executable, str(path)], check=True)
        print(f"Completed: {name}")
    except subprocess.CalledProcessError:
        print(f"Failed: {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    print("==========================================")
    print(" Inventory Recommendation System START ")
    print("==========================================\n")

    # Subsystem 1
    run_step(
        "Subsystem 1: Demand Forecast",
        BASE_DIR / "subsystem1_python_files" / "subsystem1_runner_v2.py"
    )

    # Subsystem 2
    run_step(
        "Subsystem 2: Social Trend Detection",
        BASE_DIR / "subsystem2_python_files" / "subsystem2_trend_v2.py"
    )

    # Subsystem 3
    run_step(
        "Subsystem 3: Risk Scoring",
        BASE_DIR / "subsystem3_python_files" / "risk_scoring_subsystem_v2.py"
    )

    # Subsystem 4
    run_step(
        "Subsystem 4: Final Recommendation Engine",
        BASE_DIR / "subsystem4_python_files" / "final_recommendation_v2.py"
    )

    print("\n==========================================")
    print(" SYSTEM EXECUTION COMPLETED SUCCESSFULLY ")
    print("==========================================\n")

    print(" Output:")
    print("→ subsystem4_python_files/data/output/final_recommendations_v2.csv")