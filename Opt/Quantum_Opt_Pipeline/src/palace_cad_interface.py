import json
import os
import shutil
import subprocess
from pathlib import Path
from collections.abc import Callable
import numpy as np
import pandas as pd

# Prefer an explicit environment setting, then a Palace executable on PATH.
DEFAULT_PALACE_PATH = os.environ.get("PALACE_BIN", shutil.which("palace") or "palace")

# Physical Constants
H_PLANCK = 6.62607015e-34  # J*s
E_CHARGE = 1.602176634e-19  # C
PHI_0 = 2.067833848e-15    # Wb
MAX_PALACE_MPI_PROCS = 15


def _validate_mpi_procs(n_procs: int) -> None:
    if not isinstance(n_procs, int) or not 1 <= n_procs <= MAX_PALACE_MPI_PROCS:
        raise ValueError(
            f"Palace supports between 1 and {MAX_PALACE_MPI_PROCS} MPI processes; "
            f"got {n_procs}"
        )


def update_qiskit_geometry(
    design, 
    params: np.ndarray, 
    param_keys: list[str], 
    default_unit: str = "um"
) -> None:
    """
    Dynamically updates Qiskit Metal component parameters using dot-notation:
      - 'Q1.pad_width' -> design.components['Q1'].options['pad_width']
      - 'Resonator.options.fillet' -> design.components['Resonator'].options['fillet']
      - 'lj' / 'l_j' -> bypassed (handled as lumped circuit elements)
    """
    if design is None or not hasattr(design, "components"):
        raise ValueError("A live Qiskit Metal design is required for geometry updates")

    param_dict = dict(zip(param_keys, params))

    for key, value in param_dict.items():
        # Skip circuit/lumped parameters
        if key.lower() in ["lj", "l_j", "inductance"]:
            continue

        if "." not in key:
            raise ValueError(
                f"Parameter key '{key}' must use dot notation: '<component_name>.<option_name>'"
            )

        parts = key.split(".")
        comp_name = parts[0]
        opt_path = parts[1:]

        # Strip explicit 'options' if user included it in string
        if opt_path[0] == "options":
            opt_path = opt_path[1:]

        if comp_name not in design.components:
            raise KeyError(f"Component '{comp_name}' not found in Qiskit Metal design")

        comp = design.components[comp_name]
        target_dict = comp.options

        # Traverse nested options dictionaries
        for sub_key in opt_path[:-1]:
            if sub_key in target_dict and isinstance(target_dict[sub_key], dict):
                target_dict = target_dict[sub_key]
            else:
                raise KeyError(f"Option path '{key}' does not exist on component '{comp_name}'")

        final_key = opt_path[-1]

        # Format with unit string for Qiskit Metal
        if isinstance(value, (int, float, np.floating)):
            formatted_val = f"{value:.4f}{default_unit}"
        else:
            formatted_val = str(value)

        target_dict[final_key] = formatted_val

    # Rebuild geometry layout
    design.rebuild()


def export_to_gmsh(
    design,
    output_msh: str,
    mesh_exporter: Callable[[object, str], object] | None = None,
) -> str:
    """
    Renders Qiskit Metal geometry to a conformal 3D mesh via Gmsh.
    Assigns physical tags (Vacuum, Substrate, PEC, Ports) for AWS Palace.
    """
    out_path = Path(output_msh)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if design is None:
        raise ValueError("A live Qiskit Metal design is required to export a mesh")
    if mesh_exporter is not None:
        exported_path = mesh_exporter(design, str(out_path))
    else:
        design_exporter = getattr(design, "export_to_gmsh", None)
        if not callable(design_exporter):
            raise RuntimeError(
                "No Qiskit Metal to Gmsh exporter is configured. Pass mesh_exporter="
                "(design, output_path) -> path, or provide design.export_to_gmsh."
            )
        exported_path = design_exporter(str(out_path))
    if exported_path is not None and Path(exported_path) != out_path:
        Path(exported_path).replace(out_path)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"Mesh exporter did not create {out_path}"
        )
    mesh_text = out_path.read_text(errors="replace")
    if "$Nodes" not in mesh_text or "$Elements" not in mesh_text:
        raise RuntimeError(f"{out_path} is not a populated Gmsh mesh")

    return str(out_path)


def create_palace_config(
    mesh_path: str, 
    output_dir: str, 
    sim_type: str = "Electrostatic", 
    epsilon_r: float = 11.45
) -> str:
    """
    Writes the headless JSON configuration file for AWS Palace execution.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_file = str(Path(output_dir) / "config.json")

    config = {
        "Problem": {
            "Type": sim_type,
            "Verbose": 1,
            "Output": output_dir
        },
        "Model": {
            "Mesh": mesh_path,
            "L0": 1.0e-3  # Units in mm
        },
        "Domains": {
            "Materials": [
                {"Attributes": [1], "Permittivity": 1.0},
                {"Attributes": [2], "Permittivity": epsilon_r}
            ]
        },
        "Boundaries": {
            "PEC": {"Attributes": [3, 4]},
            "Capacitance": [
                {"Index": 1, "Attributes": [5]}
            ]
        },
        "Solver": {
            "Electrostatic": {
                "Tol": 1.0e-6,
                "MaxIts": 250
            },
            "Linear": {
                "Type": "AMS",
                "KSPType": "CG"
            }
        }
    }

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    return config_file


def run_palace_solver(
    config_file: str, 
    n_procs: int = 4, 
    palace_bin: str = DEFAULT_PALACE_PATH
) -> bool:
    """
    Runs AWS Palace via MPI subprocess using the configured binary path.
    """
    _validate_mpi_procs(n_procs)
    cmd = ["mpirun", "-n", str(n_procs), palace_bin, config_file]
    
    run_dir = Path(config_file).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "palace.stdout.log"
    stderr_path = run_dir / "palace.stderr.log"
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        stdout_path.write_text(result.stdout)
        stderr_path.write_text(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"Palace failed with exit code {result.returncode}; see {stderr_path}"
            )
        return True
    except FileNotFoundError as exc:
        raise RuntimeError(f"Palace executable or mpirun was not found: {exc}") from exc


def parse_palace_results(output_dir: str, lj_val: float) -> tuple[float, float]:
    """
    Parses AWS Palace port-C.csv capacitance output and computes Ej and Ec in MHz.
    """
    c_mat_file = Path(output_dir) / "port-C.csv"

    if not c_mat_file.exists():
        raise FileNotFoundError(f"Palace did not produce {c_mat_file}")
    df = pd.read_csv(c_mat_file, header=None)
    numeric_values = pd.to_numeric(df.stack(), errors="coerce").dropna()
    if numeric_values.empty:
        raise ValueError(f"No numeric capacitance value found in {c_mat_file}")
    c_sigma = abs(float(numeric_values.iloc[0])) * 1e-15
    if not np.isfinite(c_sigma) or c_sigma <= 0:
        raise ValueError(f"Invalid capacitance value in {c_mat_file}")
    if not np.isfinite(lj_val) or lj_val <= 0:
        raise ValueError("lj_val must be a positive finite inductance")

    # Ec = e^2 / (2 * C_sigma * h) [MHz]
    ec_mhz = (E_CHARGE**2 / (2.0 * c_sigma * H_PLANCK)) * 1e-6

    # Ej = Phi_0^2 / (4 * pi^2 * Lj * h) [MHz]
    ej_mhz = (PHI_0**2 / (4.0 * (np.pi**2) * lj_val * H_PLANCK)) * 1e-6

    return float(ej_mhz), float(ec_mhz)
