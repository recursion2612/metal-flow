"""
Quantum Chip Parameter Optimization Library
Modules for AWS Palace interfacing, PhysicsNeMo surrogate modeling, and real-valued GA optimization.
"""

from .palace import (
    update_qiskit_geometry,
    export_to_gmsh,
    create_palace_config,
    run_palace_solver,
    parse_palace_results,
)
from .surrogate import (
    PhysicsNeMoSurrogate,
    deduplicate_samples,
    find_sample_logs,
)
from .genetic_algorithm import (
    compute_cost,
    select_parents,
    crossover_sbx,
    mutate_polynomial,
    produce_next_generation,
)
from .cad import (
    SqdmetalCapacitanceRunner,
    create_transmon_design,
    export_qiskit_metal_gmsh,
    extract_transmon_c_sigma,
    extract_transmon_hamiltonian_lom,
)
from .inverse_design import (
    parse_frequency,
    parse_energy,
    parse_coupling,
    parse_time,
    parse_range_or_bound,
    parse_target_params,
    parse_selection_criteria,
    invert_hamiltonian,
    compute_purcell_t1,
    compute_inverse_cost,
)

__all__ = [
    "update_qiskit_geometry",
    "export_to_gmsh",
    "create_palace_config",
    "run_palace_solver",
    "parse_palace_results",
    "PhysicsNeMoSurrogate",
    "deduplicate_samples",
    "find_sample_logs",
    "compute_cost",
    "select_parents",
    "crossover_sbx",
    "mutate_polynomial",
    "produce_next_generation",
    "SqdmetalCapacitanceRunner",
    "create_transmon_design",
    "export_qiskit_metal_gmsh",
    "extract_transmon_c_sigma",
    "extract_transmon_hamiltonian_lom",
    "parse_frequency",
    "parse_energy",
    "parse_coupling",
    "parse_time",
    "parse_range_or_bound",
    "parse_target_params",
    "parse_selection_criteria",
    "invert_hamiltonian",
    "compute_purcell_t1",
    "compute_inverse_cost",
]

