"""Train or resume the PhysicsNeMo surrogate from Palace measurements."""

import argparse

from src.nemo_surrogate import PhysicsNeMoSurrogate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-log",
        default="training_data/active_learning_log.csv",
        help="CSV containing Palace measurements",
    )
    parser.add_argument(
        "--checkpoint",
        default="training_data/checkpoints/nemo_surrogate.mdlus",
        help="PhysicsNeMo .mdlus checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    surrogate = PhysicsNeMoSurrogate(
        n_features=4,
        data_log_path=args.data_log,
        checkpoint_path=args.checkpoint,
    )
    if len(surrogate.X_train) < 5:
        raise SystemExit(
            f"Need at least 5 valid Palace samples; found {len(surrogate.X_train)}"
        )
    surrogate.fit(epochs=args.epochs)
    print(f"Trained on {len(surrogate.X_train)} measured samples")
    print(f"PhysicsNeMo model: {args.checkpoint}")
    print(f"Training state: {surrogate.optimizer_state_path}")


if __name__ == "__main__":
    main()
