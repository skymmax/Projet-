# run_all.py
"""
Master script to run the full pipeline:

1) Prepare features
2) Run baselines (Equal-Weight, Markowitz)
3) Train PPO agent
4) Evaluate PPO vs baselines
5) Run Random Forest supervised strategy
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_step(script_name: str, description: str) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print(f"Running: {script_name}")
    print("=" * 80)

    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run([sys.executable, str(script_path)], check=True)

    if result.returncode == 0:
        print(f"{description} completed successfully.")
    else:
        print(f"{description} finished with return code {result.returncode}.")


def main():
    # 1) Data preparation & EDA (optional but useful)
    run_step("run_prepare_features.py", "Prepare technical features")
    run_step("run_eda.py", "Run exploratory data analysis")

    # 2) Baseline strategies
    run_step("run_baseline.py", "Run Equal-Weight baseline")
    run_step("run_markowitz.py", "Run Markowitz minimum variance portfolio")

    # 3) Reinforcement Learning (PPO)
    run_step("run_train_ppo.py", "Train PPO agent")
    run_step("run_evaluate_ppo.py", "Evaluate PPO vs Equal-Weight baseline")

    # 4) Supervised learning baseline (Random Forest)
    run_step("run_random_forest.py", "Run Random Forest supervised strategy")
    
    run_step("run_evaluate_ppo.py", "Evaluate PPO vs baselines and save allocation plots")



if __name__ == "__main__":
    main()
