# run_all.py

# Master script to run the full pipeline:
# 1) Prepare features
# 2) Run baselines (Equal-Weight, Markowitz)
# 3) Run Random Forest supervised strategy
# 4) Train and evaluate PPO agent

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_step(script_name: str, description: str) -> None:
    """Run a single script from the scripts/ folder."""
    print(f"STEP: {description}")
    print(f"Running: {script_name}")

    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run([sys.executable, str(script_path)], check=True)

    if result.returncode == 0:
        print(f" {description} completed successfully")
    else:
        print(f" {description} finished with return code {result.returncode}")


def main() -> None:
    # 1) Data preparation
    run_step("run_prepare.py", "Prepare technical features")

    # 2) Baseline strategies (Equal-Weight and Markowitz)
    run_step("run_baselines.py", "Run classical baselines")

    # 3) Supervised learning baseline (Random Forest)
    run_step("run_random_forest.py", "Run Random Forest supervised strategy")

    # 4) Reinforcement Learning (PPO: train + evaluate)
    run_step("run_ppo.py", "Train and evaluate PPO agent")


if __name__ == "__main__":
    main()
