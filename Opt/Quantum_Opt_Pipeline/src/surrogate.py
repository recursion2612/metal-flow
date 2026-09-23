from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


try:
    import physicsnemo
except ImportError as exc:
    physicsnemo = None
    _PHYSICSNEMO_IMPORT_ERROR = exc
else:
    _PHYSICSNEMO_IMPORT_ERROR = None


_PhysicsNeMoModule = physicsnemo.Module if physicsnemo is not None else nn.Module


class MeshGraphNet(_PhysicsNeMoModule):
    """
    MeshGraphNet-style surrogate for compact geometry graphs.

    The project's design data is parameter-vector based, so this converts each
    scalar parameter into a node in a tiny ring graph. That preserves the current
    CSV-based API while moving the surrogate to a mesh-graph formulation.
    """

    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.node_encoder = nn.Linear(2, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_features),
        )

    def _to_graph(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"Expected graph tensor with shape (batch, features), got {tuple(x.shape)}")

        batch_size, n_features = x.shape
        if n_features == 0:
            raise ValueError("Graph node feature dimension cannot be zero")

        node_ids = torch.arange(n_features, dtype=x.dtype, device=x.device)
        node_ids = (node_ids / max(n_features - 1, 1)).view(1, n_features, 1)
        graph_x = torch.cat([x.unsqueeze(-1), node_ids.expand(batch_size, -1, -1)], dim=-1)
        edge_src = []
        edge_dst = []
        n_nodes = n_features
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            edge_src.extend([i, j])
            edge_dst.extend([j, i])
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=x.device)
        return graph_x, edge_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        graph_x, edge_index = self._to_graph(x)
        node_states = self.node_encoder(graph_x)

        for _ in range(self.n_layers):
            message_buffer = torch.zeros_like(node_states)
            for edge_idx in range(edge_index.shape[1]):
                src = int(edge_index[0, edge_idx].item())
                dst = int(edge_index[1, edge_idx].item())
                aggregated = torch.cat([node_states[:, src, :], node_states[:, dst, :]], dim=-1)
                message = self.edge_encoder(aggregated)
                message_buffer[:, dst, :] = message_buffer[:, dst, :] + message

            node_update_input = torch.cat([node_states, message_buffer], dim=-1)
            node_states = self.node_update(node_update_input)
            node_states = F.silu(node_states)

        graph_embedding = node_states.mean(dim=1)
        return self.readout(graph_embedding)


QuantumSurrogateNet = MeshGraphNet


def random_seed() -> int:
    """Return a fresh process seed from NumPy's entropy source."""
    return int(np.random.SeedSequence().generate_state(1)[0])


class PhysicsNeMoSurrogate:
    """
    Surrogate management class handling:
      1. Dynamic hardware acceleration (MPS, CUDA, or CPU)
      2. Persistent dataset logging to CSV
    3. PhysicsNeMo checkpointing and backpropagation training
      4. Monte Carlo Dropout for epistemic uncertainty estimation
    """
    TARGET_TRANSFORM = "log_ej_v1"
    INPUT_TRANSFORM = "log_lj_v1"
    MODEL_VERSION = "typed_parameter_nodes_dropout_v2"

    def __init__(
        self, 
        n_features: int, 
        data_log_path: str = "training_data/active_learning_log.csv",
        checkpoint_path: str | None = None,
    ):
        self.n_features = n_features
        self.data_log_path = Path(data_log_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (
            self.data_log_path.parent / "checkpoints" / "nemo_surrogate.mdlus"
        )
        self.optimizer_state_path = self.checkpoint_path.with_suffix(".state.pt")
        self.evaluation_path = self.checkpoint_path.with_suffix(".evaluation.json")

        if physicsnemo is None:
            raise ImportError(
                "NVIDIA PhysicsNeMo is required for the surrogate. Install "
                "nvidia-physicsnemo before starting optimization."
            ) from _PHYSICSNEMO_IMPORT_ERROR
        
        # 1. Device Selection: Apple Silicon MPS -> NVIDIA CUDA -> CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.model = QuantumSurrogateNet(in_features=n_features).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        
        self.is_trained = False
        self.last_evaluation = None
        self.X_train = np.empty((0, n_features), dtype=np.float64)
        self.Y_train = np.empty((0, 2), dtype=np.float64)
        
        # Normalization bounds placeholders
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        
        self._init_data_log()
        self._load_data_log()
        self._load_checkpoint()
        if self.evaluation_path.exists():
            try:
                self.last_evaluation = json.loads(
                    self.evaluation_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Unable to load surrogate evaluation {self.evaluation_path}: {exc}"
                ) from exc

    def _init_data_log(self) -> None:
        """Creates the training CSV file with appropriate column headers if missing."""
        self.data_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.data_log_path.exists():
            headers = [f"param_{i}" for i in range(self.n_features)] + ["Ej_MHz", "Ec_MHz"]
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.data_log_path, index=False)

    def _load_data_log(self) -> None:
        """Restore valid Palace samples before the first prediction."""
        try:
            df = pd.read_csv(self.data_log_path)
            expected_columns = [f"param_{i}" for i in range(self.n_features)] + ["Ej_MHz", "Ec_MHz"]
            if list(df.columns) != expected_columns:
                raise ValueError(f"expected columns {expected_columns}, got {list(df.columns)}")
            values = df.to_numpy(dtype=np.float64)
            values = values[np.all(np.isfinite(values), axis=1)]
            if values.size:
                self.X_train = values[:, :self.n_features]
                self.Y_train = values[:, self.n_features:]
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Unable to load surrogate data from {self.data_log_path}: {exc}") from exc

    def _load_checkpoint(self) -> None:
        """Resume a compatible model; the CSV remains the source of training samples."""
        if not self.checkpoint_path.exists():
            return
        try:
            if not self.optimizer_state_path.exists():
                return
            checkpoint = torch.load(
                self.optimizer_state_path,
                map_location="cpu",
                weights_only=False,
            )
            if (
                checkpoint.get("target_transform") != self.TARGET_TRANSFORM
                or checkpoint.get("input_transform") != self.INPUT_TRANSFORM
                or checkpoint.get("model_version") != self.MODEL_VERSION
            ):
                return
            self.model.load(self.checkpoint_path, map_location="cpu")
            if checkpoint.get("n_features") != self.n_features:
                return
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.x_min = np.asarray(checkpoint["x_min"], dtype=np.float64)
            self.x_max = np.asarray(checkpoint["x_max"], dtype=np.float64)
            self.y_min = np.asarray(checkpoint["y_min"], dtype=np.float64)
            self.y_max = np.asarray(checkpoint["y_max"], dtype=np.float64)
            self.is_trained = bool(checkpoint.get("is_trained", False)) and len(self.X_train) >= 5
            self.model.to(self.device)
        except (OSError, KeyError, RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Unable to load surrogate checkpoint {self.checkpoint_path}: {exc}") from exc

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.checkpoint_path)
        torch.save({
            "n_features": self.n_features,
            "optimizer_state": self.optimizer.state_dict(),
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "is_trained": self.is_trained,
            "target_transform": self.TARGET_TRANSFORM,
            "input_transform": self.INPUT_TRANSFORM,
            "model_version": self.MODEL_VERSION,
        }, self.optimizer_state_path)
        if self.last_evaluation is not None:
            self.evaluation_path.write_text(
                json.dumps(self.last_evaluation, indent=2), encoding="utf-8"
            )

    def log_and_append_sample(self, x: np.ndarray, y: list[float] | np.ndarray) -> None:
        """
        Saves a validated Palace simulation result to in-memory memory buffers 
        and appends it to the disk log for persistence across sessions.
        """
        x_reshaped = np.array(x, dtype=np.float64).reshape(1, -1)
        y_reshaped = np.array(y, dtype=np.float64).reshape(1, -1)
        if x_reshaped.shape[1] != self.n_features or y_reshaped.shape[1] != 2:
            raise ValueError("surrogate samples must have n_features inputs and two targets")
        if not np.all(np.isfinite(x_reshaped)) or not np.all(np.isfinite(y_reshaped)):
            raise ValueError("surrogate samples must contain only finite values")

        self.X_train = np.vstack([self.X_train, x_reshaped]) if self.X_train.size else x_reshaped
        self.Y_train = np.vstack([self.Y_train, y_reshaped]) if self.Y_train.size else y_reshaped

        record = np.hstack([x_reshaped, y_reshaped])
        df = pd.DataFrame(record)
        df.to_csv(self.data_log_path, mode="a", header=False, index=False)

    def _train_arrays(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int,
        validation_X: np.ndarray | None = None,
        validation_Y: np.ndarray | None = None,
        patience: int = 200,
    ) -> int:
        x_transformed = self._transform_inputs(X)
        x_norm = (x_transformed - self.x_min) / (self.x_max - self.x_min)
        y_transformed = self._transform_targets(Y)
        y_norm = (y_transformed - self.y_min) / (self.y_max - self.y_min)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32).to(self.device)
        validation_tensor = None
        validation_target = None
        if validation_X is not None and validation_Y is not None:
            validation_transformed = self._transform_inputs(validation_X)
            validation_tensor = torch.tensor(
                (validation_transformed - self.x_min) / (self.x_max - self.x_min),
                dtype=torch.float32,
            ).to(self.device)
            validation_target = torch.tensor(
                (self._transform_targets(validation_Y) - self.y_min)
                / (self.y_max - self.y_min),
                dtype=torch.float32,
            ).to(self.device)

        self.model.train()
        best_state = None
        best_loss = float("inf")
        stale_epochs = 0
        epochs_run = 0
        for _ in range(epochs):
            self.optimizer.zero_grad()
            preds = self.model(x_tensor)
            loss = self.criterion(preds, y_tensor)
            loss.backward()
            self.optimizer.step()
            epochs_run += 1
            if validation_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    validation_loss = float(
                        self.criterion(
                            self.model(validation_tensor), validation_target
                        ).item()
                    )
                self.model.train()
                if validation_loss < best_loss - 1e-7:
                    best_loss = validation_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return epochs_run

    @staticmethod
    def _transform_inputs(X: np.ndarray) -> np.ndarray:
        transformed = np.asarray(X, dtype=np.float64).copy()
        if np.any(transformed[:, 3] <= 0):
            raise ValueError("lj inputs must be positive for log-space features")
        transformed[:, 3] = np.log(transformed[:, 3])
        return transformed

    @staticmethod
    def _transform_targets(Y: np.ndarray) -> np.ndarray:
        transformed = np.asarray(Y, dtype=np.float64).copy()
        if np.any(transformed[:, 0] <= 0):
            raise ValueError("Ej targets must be positive for log-space training")
        transformed[:, 0] = np.log(transformed[:, 0])
        return transformed

    @staticmethod
    def _inverse_transform_targets(Y: np.ndarray) -> np.ndarray:
        restored = np.asarray(Y, dtype=np.float64).copy()
        restored[:, 0] = np.exp(restored[:, 0])
        return restored

    def _score(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        target_thresholds: np.ndarray,
        accuracy_tolerance_percent: float,
    ) -> dict[str, float]:
        x_transformed = self._transform_inputs(X)
        x_norm = (x_transformed - self.x_min) / (self.x_max - self.x_min)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                torch.tensor(x_norm, dtype=torch.float32).to(self.device)
            ).cpu().numpy()
        predictions = self._inverse_transform_targets(
            predictions * (self.y_max - self.y_min) + self.y_min
        )
        errors = predictions - Y
        absolute_errors = np.abs(errors)
        relative_errors = absolute_errors / np.maximum(np.abs(Y), 1e-8)
        metrics = {
            "mse": float(np.mean(errors ** 2)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(absolute_errors)),
            "ej_mae_mhz": float(np.mean(absolute_errors[:, 0])),
            "ec_mae_mhz": float(np.mean(absolute_errors[:, 1])),
            "ej_accuracy_percent": float(
                np.mean(relative_errors[:, 0] <= accuracy_tolerance_percent / 100) * 100
            ),
            "ec_accuracy_percent": float(
                np.mean(relative_errors[:, 1] <= accuracy_tolerance_percent / 100) * 100
            ),
        }
        for index, name in enumerate(("ej", "ec")):
            actual_positive = Y[:, index] >= target_thresholds[index]
            predicted_positive = predictions[:, index] >= target_thresholds[index]
            true_positive = np.sum(actual_positive & predicted_positive)
            false_positive = np.sum(~actual_positive & predicted_positive)
            false_negative = np.sum(actual_positive & ~predicted_positive)
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            metrics[f"{name}_precision"] = float(precision)
            metrics[f"{name}_recall"] = float(recall)
            metrics[f"{name}_f1"] = float(
                2 * precision * recall / max(precision + recall, 1e-12)
            )
        return metrics

    def fit(
        self,
        epochs: int = 150,
        validation_split: float = 0.2,
        evaluation_seed: int | None = None,
        accuracy_tolerance_percent: float = 5.0,
        target_thresholds: tuple[float, float] | None = None,
        early_stopping_patience: int = 200,
    ) -> dict:
        """
        Normalizes dataset inputs/targets and trains the neural network.
        Requires at least 5 simulation points to avoid overfitting trivial solutions.
        """
        if len(self.X_train) < 5:
            return {"status": "insufficient_data", "sample_count": len(self.X_train)}

        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if accuracy_tolerance_percent <= 0:
            raise ValueError("accuracy_tolerance_percent must be greater than 0")
        if early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")

        sample_count = len(self.X_train)
        validation_count = int(sample_count * validation_split)
        if validation_split and validation_count == 0:
            validation_count = 1
        if sample_count - validation_count < 2:
            raise ValueError("validation_split leaves fewer than 2 training samples")

        indices = np.random.default_rng(evaluation_seed).permutation(sample_count)
        validation_indices = indices[:validation_count]
        training_indices = indices[validation_count:]

        # Min-Max Normalization to stable [0, 1] range
        fit_X = self.X_train[training_indices]
        fit_Y = self.Y_train[training_indices]
        thresholds = np.asarray(
            target_thresholds if target_thresholds is not None else np.median(fit_Y, axis=0),
            dtype=np.float64,
        )
        if thresholds.shape != (2,) or not np.all(np.isfinite(thresholds)):
            raise ValueError("target_thresholds must contain two finite values")
        transformed_fit_X = self._transform_inputs(fit_X)
        self.x_min, x_max = transformed_fit_X.min(axis=0), transformed_fit_X.max(axis=0)
        transformed_fit_Y = self._transform_targets(fit_Y)
        self.y_min, y_max = transformed_fit_Y.min(axis=0), transformed_fit_Y.max(axis=0)
        self.x_max = self.x_min + np.maximum(x_max - self.x_min, 1e-8)
        self.y_max = self.y_min + np.maximum(y_max - self.y_min, 1e-8)

        epochs_run = self._train_arrays(
            fit_X,
            fit_Y,
            epochs,
            validation_X=self.X_train[validation_indices] if validation_count else None,
            validation_Y=self.Y_train[validation_indices] if validation_count else None,
            patience=early_stopping_patience,
        )
        validation_metrics = self._score(
            self.X_train[validation_indices],
            self.Y_train[validation_indices],
            thresholds,
            accuracy_tolerance_percent,
        ) if validation_count else None

        # The final checkpoint uses every measured sample after the unbiased score.
        transformed_X = self._transform_inputs(self.X_train)
        self.x_min = transformed_X.min(axis=0)
        transformed_Y = self._transform_targets(self.Y_train)
        self.y_min = transformed_Y.min(axis=0)
        self.x_max = self.x_min + np.maximum(self.X_train.max(axis=0) - self.x_min, 1e-8)
        self.y_max = self.y_min + np.maximum(transformed_Y.max(axis=0) - self.y_min, 1e-8)
        self._train_arrays(self.X_train, self.Y_train, epochs)

        self.is_trained = True
        train_metrics = self._score(
            self.X_train,
            self.Y_train,
            thresholds,
            accuracy_tolerance_percent,
        )
        self.last_evaluation = {
            "status": "trained",
            "sample_count": sample_count,
            "training_samples": int(len(training_indices)),
            "validation_samples": validation_count,
            "epochs": epochs,
            "epochs_run": epochs_run,
            "early_stopping_patience": early_stopping_patience,
            "validation_split": validation_split,
            "evaluation_seed": evaluation_seed,
            "accuracy_tolerance_percent": accuracy_tolerance_percent,
            "target_thresholds": thresholds.tolist(),
            "target_transform": self.TARGET_TRANSFORM,
            "input_transform": self.INPUT_TRANSFORM,
            "model_version": self.MODEL_VERSION,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        self._save_checkpoint()
        return self.last_evaluation

    def evaluate_log(
        self,
        data_log_path: str,
        target_thresholds: tuple[float, float] | None = None,
        accuracy_tolerance_percent: float = 5.0,
    ) -> dict:
        """Evaluate a frozen checkpoint against an independent CSV dataset."""
        if not self.is_trained or self.x_min is None:
            raise RuntimeError("evaluate_log requires a trained surrogate checkpoint")
        frame = pd.read_csv(data_log_path)
        expected_columns = [f"param_{i}" for i in range(self.n_features)] + ["Ej_MHz", "Ec_MHz"]
        if list(frame.columns) != expected_columns:
            raise ValueError(f"expected columns {expected_columns}, got {list(frame.columns)}")
        values = frame.to_numpy(dtype=np.float64)
        values = values[np.all(np.isfinite(values), axis=1)]
        if len(values) == 0:
            raise ValueError(f"No finite samples found in {data_log_path}")
        thresholds = np.asarray(
            target_thresholds
            if target_thresholds is not None
            else self.last_evaluation["target_thresholds"],
            dtype=np.float64,
        )
        metrics = self._score(
            values[:, :self.n_features],
            values[:, self.n_features:],
            thresholds,
            accuracy_tolerance_percent,
        )
        return {"sample_count": len(values), "data_log": data_log_path, **metrics}

    def predict(self, X: np.ndarray, n_mc_samples: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """
        Inference with Monte Carlo Dropout:
        Runs multiple forward passes with active dropout layers to quantify epistemic uncertainty.

        Returns:
            mean_preds: np.ndarray of shape (N, 2) with predicted [Ej, Ec] in MHz
            uncertainty: np.ndarray of shape (N,) with the standard deviation across samples
        """
        n_candidates = len(X)
        
        # Cold start fallback: force high uncertainty to trigger full Palace runs
        if not self.is_trained or len(self.X_train) < 5:
            default_preds = np.tile([22000.0, 400.0], (n_candidates, 1))
            return default_preds, np.full(n_candidates, 999.0)

        # Normalize incoming candidate parameters
        x_transformed = self._transform_inputs(X)
        x_norm = (x_transformed - self.x_min) / (self.x_max - self.x_min)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(self.device)

        # Enable dropout during inference to sample posterior distributions
        self.model.train()
        mc_predictions = []

        with torch.no_grad():
            for _ in range(n_mc_samples):
                pred_norm = self.model(x_tensor).cpu().numpy()
                pred_real = self._inverse_transform_targets(
                    pred_norm * (self.y_max - self.y_min) + self.y_min
                )
                mc_predictions.append(pred_real)

        mc_predictions = np.array(mc_predictions)  # Shape: (n_mc_samples, N, 2)
        mean_preds = np.mean(mc_predictions, axis=0)
        
        # Uncertainty metric: average standard deviation across Ej and Ec
        uncertainty = np.mean(np.std(mc_predictions, axis=0), axis=1)
        return mean_preds, uncertainty
