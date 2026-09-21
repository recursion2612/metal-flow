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
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for an unbiased evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=42,
        help="Seed used to select the validation samples",
    )
    parser.add_argument(
        "--accuracy-tolerance-percent",
        type=float,
        default=5.0,
        help="Relative error allowed for an accurate prediction (default: 5%%)",
    )
    parser.add_argument("--ej-threshold-mhz", type=float, help="Ej threshold for precision/F1")
    parser.add_argument("--ec-threshold-mhz", type=float, help="Ec threshold for precision/F1")
    parser.add_argument(
        "--evaluation-output",
        help="Optional JSON path for the training and validation report",
    )
    args = parser.parse_args()
    if (args.ej_threshold_mhz is None) != (args.ec_threshold_mhz is None):
        parser.error("--ej-threshold-mhz and --ec-threshold-mhz must be provided together")

    surrogate = PhysicsNeMoSurrogate(
        n_features=4,
        data_log_path=args.data_log,
        checkpoint_path=args.checkpoint,
    )
    if len(surrogate.X_train) < 5:
        raise SystemExit(
            f"Need at least 5 valid Palace samples; found {len(surrogate.X_train)}"
        )
    evaluation = surrogate.fit(
        epochs=args.epochs,
        validation_split=args.validation_split,
        evaluation_seed=args.evaluation_seed,
        accuracy_tolerance_percent=args.accuracy_tolerance_percent,
        target_thresholds=(args.ej_threshold_mhz, args.ec_threshold_mhz)
        if args.ej_threshold_mhz is not None and args.ec_threshold_mhz is not None
        else None,
    )
    if args.evaluation_output:
        from pathlib import Path
        import json

        output_path = Path(args.evaluation_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Trained on {len(surrogate.X_train)} measured samples")
    print(f"PhysicsNeMo model: {args.checkpoint}")
    print(f"Training state: {surrogate.optimizer_state_path}")
    print(f"Model status: {evaluation['status']}")
    if evaluation.get("validation_metrics"):
        metrics = evaluation["validation_metrics"]
        print(f"Validation MAE: Ej={metrics['ej_mae_mhz']:.3f} MHz, Ec={metrics['ec_mae_mhz']:.3f} MHz")
        print(
            "Validation accuracy: "
            f"Ej={metrics['ej_accuracy_percent']:.1f}%, "
            f"Ec={metrics['ec_accuracy_percent']:.1f}%"
        )
        print(
            "Validation precision/F1: "
            f"Ej={metrics['ej_precision']:.3f}/{metrics['ej_f1']:.3f}, "
            f"Ec={metrics['ec_precision']:.3f}/{metrics['ec_f1']:.3f}"
        )
    if args.evaluation_output:
        print(f"Evaluation report: {args.evaluation_output}")


if __name__ == "__main__":
    main()
