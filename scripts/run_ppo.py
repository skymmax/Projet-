# scripts/run_ppo.py
# Train and evaluate the PPO reinforcement learning agent.
#
# Steps:
# - Train PPO on the training period
# - Evaluate PPO on the test set
#   (compare to Equal-Weight and save allocation plots)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.train_ppo import main as train_ppo_main
from src.agents.evaluate_ppo import main as eval_ppo_main


def main() -> None:
    # --- Train PPO agent ---
    print("\n--- PPO training ---")
    train_ppo_main()

    # --- Evaluate PPO agent ---
    print("\n--- PPO evaluation ---")
    eval_ppo_main()

    print("\n--- PPO pipeline finished ---")


if __name__ == "__main__":
    main()
