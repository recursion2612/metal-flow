"""Inverse design and Hamiltonian specification engine for superconducting transmon qubits.

Allows designing transmons directly from operational cQED targets:
    f_q in [4, 6] GHz
    alpha in [-350, -150] MHz
    g in [5, 50] MHz
with selection constraints on:
    T1 (Purcell coherence)
    Ej/Ec ratio (transmon charge dispersion regime)
    chi (dispersive readout shift)
    dispersion (charge noise sensitivity)
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np

from .palace import E_CHARGE, H_PLANCK, PHI_0


def parse_frequency(val: str | float | int) -> float:
    """Parse frequency string or number to GHz float.
    
    Examples:
        '4GHz' -> 4.0
        '4.5 GHz' -> 4.5
        '5000MHz' -> 5.0
        '5e9' -> 5.0
        4.2 -> 4.2
    """
    if isinstance(val, (int, float)):
        # If > 1e6 assume Hz, if > 100 assume MHz, otherwise GHz
        if val > 1e6:
            return float(val / 1e9)
        if val > 100:
            return float(val / 1e3)
        return float(val)
    
    val_str = str(val).strip().lower()
    m_ghz = re.match(r"^([\d.]+)\s*(?:ghz|g)?$", val_str)
    if m_ghz:
        return float(m_ghz.group(1))
    
    m_mhz = re.match(r"^([\d.]+)\s*mhz$", val_str)
    if m_mhz:
        return float(m_mhz.group(1)) / 1000.0
    
    m_hz = re.match(r"^([\d.e+-]+)\s*hz$", val_str)
    if m_hz:
        return float(m_hz.group(1)) / 1e9
    
    try:
        raw_val = float(val_str)
        return parse_frequency(raw_val)
    except ValueError as exc:
        raise ValueError(f"Unable to parse frequency string: '{val}'") from exc


def parse_energy(val: str | float | int) -> float:
    """Parse energy or anharmonicity string to MHz float.
    
    Examples:
        '200 MHz' -> 200.0
        '-220MHz' -> 220.0 (absolute value)
        '0.2 GHz' -> 200.0
        250.0 -> 250.0
    """
    if isinstance(val, (int, float)):
        if abs(val) > 1e6:
            return abs(float(val / 1e6))
        if abs(val) < 1.0:  # assume GHz
            return abs(float(val * 1000.0))
        return abs(float(val))
    
    val_str = str(val).strip().lower().lstrip("-")
    m_mhz = re.match(r"^([\d.]+)\s*(?:mhz|m)?$", val_str)
    if m_mhz:
        return float(m_mhz.group(1))
    
    m_ghz = re.match(r"^([\d.]+)\s*ghz$", val_str)
    if m_ghz:
        return float(m_ghz.group(1)) * 1000.0
    
    m_hz = re.match(r"^([\d.e+-]+)\s*hz$", val_str)
    if m_hz:
        return float(m_hz.group(1)) / 1e6
    
    try:
        raw_val = float(val_str)
        return parse_energy(raw_val)
    except ValueError as exc:
        raise ValueError(f"Unable to parse energy string: '{val}'") from exc


def parse_coupling(val: str | float | int) -> float:
    """Parse coupling rate g to MHz float.
    
    Examples:
        '10 MHz' -> 10.0
        '0.01 GHz' -> 10.0
        15.0 -> 15.0
    """
    return parse_energy(val)


def parse_time(val: str | float | int) -> float:
    """Parse time string or number to microseconds (us) float.
    
    Examples:
        '>100us' -> 100.0
        '100 us' -> 100.0
        '0.1 ms' -> 100.0
        '1e-4 s' -> 100.0
        50.0 -> 50.0
    """
    if isinstance(val, (int, float)):
        if val < 1e-3:  # assume seconds
            return float(val * 1e6)
        if val > 1000:  # assume ns
            return float(val / 1e3)
        return float(val)
    
    val_str = str(val).strip().lower().lstrip("><= ")
    m_us = re.match(r"^([\d.]+)\s*(?:us|µs|usec)?$", val_str)
    if m_us:
        return float(m_us.group(1))
    
    m_ms = re.match(r"^([\d.]+)\s*ms$", val_str)
    if m_ms:
        return float(m_ms.group(1)) * 1000.0
    
    m_ns = re.match(r"^([\d.]+)\s*ns$", val_str)
    if m_ns:
        return float(m_ns.group(1)) / 1000.0
    
    m_s = re.match(r"^([\d.e+-]+)\s*s$", val_str)
    if m_s:
        return float(m_s.group(1)) * 1e6
    
    try:
        raw_val = float(val_str)
        return parse_time(raw_val)
    except ValueError as exc:
        raise ValueError(f"Unable to parse time string: '{val}'") from exc


def parse_range_or_bound(val: Any) -> tuple[float, float]:
    """Parse range or bounds string into (min_val, max_val).
    
    Examples:
        '>50, <100' -> (50.0, 100.0)
        '>50' -> (50.0, float('inf'))
        '<100' -> (0.0, 100.0)
        '50-100' -> (50.0, 100.0)
        (50, 100) -> (50.0, 100.0)
    """
    if isinstance(val, (tuple, list)):
        return float(val[0]), float(val[1])
    if isinstance(val, (int, float)):
        return float(val), float("inf")
    
    val_str = str(val).strip().lower()
    min_v, max_v = 0.0, float("inf")
    
    # Check for comma-separated conditions e.g. '>50, <100'
    parts = [p.strip() for p in val_str.split(",")]
    if len(parts) > 1:
        for p in parts:
            if ">" in p or "min" in p:
                min_v = float(re.findall(r"[\d.]+", p)[0])
            elif "<" in p or "max" in p:
                max_v = float(re.findall(r"[\d.]+", p)[0])
        return min_v, max_v
    
    # Check single '>' or '>='
    if val_str.startswith(">"):
        num = float(re.findall(r"[\d.]+", val_str)[0])
        return num, float("inf")
    
    # Check single '<' or '<='
    if val_str.startswith("<"):
        num = float(re.findall(r"[\d.]+", val_str)[0])
        return 0.0, num
    
    # Check hyphenated '50-100'
    m_hyphen = re.match(r"^([\d.]+)\s*-\s*([\d.]+)$", val_str)
    if m_hyphen:
        return float(m_hyphen.group(1)), float(m_hyphen.group(2))
    
    # Fallback to single numeric value as lower bound
    nums = re.findall(r"[\d.]+", val_str)
    if nums:
        return float(nums[0]), float("inf")
    
    raise ValueError(f"Unable to parse range or bound from: '{val}'")


def parse_target_params(params_dict: dict[str, Any]) -> dict[str, float]:
    """Parse high-level user parameter dictionary into canonical units.
    
    Expected keys (case-insensitive):
        fQ / fq / f_q: Qubit frequency (e.g. '4GHz')
        alpha / anharmonicity: Anharmonicity magnitude (e.g. '200MHz')
        g / coupling / g_bus: Coupler coupling rate (e.g. '10MHz')
    """
    cleaned: dict[str, float] = {}
    lower_map = {str(k).lower().replace("_", ""): v for k, v in params_dict.items()}
    
    # 1. Qubit frequency
    fq_val = lower_map.get("fq") or lower_map.get("freq") or lower_map.get("frequency")
    if fq_val is None:
        raise ValueError("params_dict must specify 'fQ' (qubit frequency, e.g. '4GHz')")
    cleaned["fQ_GHz"] = parse_frequency(fq_val)
    
    # 2. Anharmonicity
    alpha_val = lower_map.get("alpha") or lower_map.get("anharmonicity") or lower_map.get("ec")
    if alpha_val is None:
        # Default typical transmon anharmonicity if omitted: 250 MHz
        cleaned["alpha_MHz"] = 250.0
    else:
        cleaned["alpha_MHz"] = parse_energy(alpha_val)
    
    # 3. Coupling rate g
    g_val = lower_map.get("g") or lower_map.get("coupling") or lower_map.get("gbus")
    if g_val is not None:
        cleaned["g_MHz"] = parse_coupling(g_val)
    else:
        cleaned["g_MHz"] = 10.0  # default 10 MHz
    
    return cleaned


def parse_selection_criteria(selection: dict[str, Any] | None) -> dict[str, Any]:
    """Parse selection criteria for the Genetic Algorithm.
    
    Supported keys:
        T1: Purcell coherence threshold (e.g. '>100us')
        ratio: Transmon Ej/Ec ratio bounds (e.g. '>50, <100')
        chi: Dispersive shift range in MHz (e.g. '>0.5MHz, <5MHz')
        dispersion: Charge dispersion upper limit (e.g. '<10kHz')
    """
    defaults = {
        "min_T1_us": 100.0,
        "min_ratio": 50.0,
        "max_ratio": 100.0,
        "min_chi_MHz": 0.5,
        "max_chi_MHz": 10.0,
        "max_dispersion_kHz": 10.0,
    }
    if not selection:
        return defaults
    
    criteria = defaults.copy()
    lower_map = {str(k).lower().replace("_", ""): v for k, v in selection.items()}
    
    # T1
    if "t1" in lower_map:
        criteria["min_T1_us"] = parse_time(lower_map["t1"])
    
    # ratio
    if "ratio" in lower_map or "ejec" in lower_map:
        min_r, max_r = parse_range_or_bound(lower_map.get("ratio") or lower_map.get("ejec"))
        criteria["min_ratio"] = min_r
        if np.isfinite(max_r):
            criteria["max_ratio"] = max_r
    
    # chi
    if "chi" in lower_map:
        min_c, max_c = parse_range_or_bound(lower_map["chi"])
        criteria["min_chi_MHz"] = min_c
        if np.isfinite(max_c):
            criteria["max_chi_MHz"] = max_c
    
    # dispersion
    if "dispersion" in lower_map:
        val = lower_map["dispersion"]
        _, max_disp = parse_range_or_bound(val)
        criteria["max_dispersion_kHz"] = max_disp
    
    return criteria


def invert_hamiltonian(f_q_GHz: float, alpha_MHz: float) -> dict[str, float]:
    """Invert target (f_q, alpha) to exact Hamiltonian parameters (Ej, Ec, Lj, C_sigma).
    
    In the transmon regime (Ej/Ec >> 1):
        Ec ~ |alpha|
        f_q ~ (sqrt(8*Ej*Ec) - Ec) / h
        => Ej = (h*f_q + Ec)^2 / (8*Ec)
        C_sigma = e^2 / (2 * Ec * h)
        Lj = Phi_0^2 / (4 * pi^2 * Ej * h)
    
    Returns:
        dict containing:
            fQ_GHz, alpha_MHz, Ej_MHz, Ec_MHz, ratio, Lj_H, Lj_nH, C_sigma_F, C_sigma_fF
    """
    if f_q_GHz <= 0 or alpha_MHz <= 0:
        raise ValueError("f_q_GHz and alpha_MHz must be strictly positive")
    
    ec_mhz = float(abs(alpha_MHz))
    fq_mhz = float(f_q_GHz * 1000.0)
    
    # Closed-form Duffing expansion inversion
    ej_mhz = float((fq_mhz + ec_mhz) ** 2 / (8.0 * ec_mhz))
    ratio = float(ej_mhz / ec_mhz)
    
    # Physical capacitance and inductance
    c_sigma = float(E_CHARGE**2 / (2.0 * (ec_mhz * 1e6) * H_PLANCK))
    lj = float(PHI_0**2 / (4.0 * np.pi**2 * (ej_mhz * 1e6) * H_PLANCK))
    
    return {
        "fQ_GHz": f_q_GHz,
        "alpha_MHz": ec_mhz,
        "Ej_MHz": ej_mhz,
        "Ec_MHz": ec_mhz,
        "ratio": ratio,
        "Lj_H": lj,
        "Lj_nH": lj * 1e9,
        "C_sigma_F": c_sigma,
        "C_sigma_fF": c_sigma * 1e15,
    }


def compute_purcell_t1(
    f_q_GHz: float,
    g_MHz: float,
    f_r_GHz: float = 7.0,
    q_readout: float = 1e4,
) -> float:
    """Calculate the Purcell-limited relaxation time T1 in microseconds."""
    wq = 2.0 * np.pi * (f_q_GHz * 1e9)
    wr = 2.0 * np.pi * (f_r_GHz * 1e9)
    g = 2.0 * np.pi * (g_MHz * 1e6)
    
    if g <= 0 or wq <= 0 or wr <= 0:
        return 100.0
    
    kbus = wr / max(1.0, q_readout)
    t1_sec = (wr**2 - wq**2) ** 2 / (4.0 * kbus * (g**2) * (wq**2))
    return float(abs(t1_sec) * 1e6)


def compute_inverse_cost(
    predicted_metrics: dict[str, float],
    target_params: dict[str, float],
    selection: dict[str, Any],
) -> float:
    """Compute multi-objective fitness loss with exterior barrier penalties.
    
    Primary objectives:
        - Relative error on fQ
        - Relative error on alpha
        - Relative error on g (if provided)
    
    Barrier penalties (selection criteria):
        - Purcell T1 < min_T1
        - Ej/Ec < min_ratio or Ej/Ec > max_ratio
        - chi < min_chi or chi > max_chi
        - dispersion > max_dispersion
    """
    fq_pred = predicted_metrics.get("fQ_GHz", 5.0)
    alpha_pred = predicted_metrics.get("alpha_MHz", 250.0)
    g_pred = predicted_metrics.get("g_MHz", 10.0)
    ratio_pred = predicted_metrics.get("ratio", fq_pred * 1000.0 / alpha_pred)
    t1_pred = predicted_metrics.get("T1_us", 150.0)
    chi_pred = predicted_metrics.get("chi_MHz", 1.5)
    disp_pred = predicted_metrics.get("dispersion_kHz", 0.1)
    
    fq_target = target_params.get("fQ_GHz", 5.0)
    alpha_target = target_params.get("alpha_MHz", 250.0)
    g_target = target_params.get("g_MHz", 10.0)
    
    # 1. Target Tracking Loss (Normalized MSE)
    loss_target = (
        0.5 * ((fq_pred - fq_target) / fq_target) ** 2
        + 0.3 * ((alpha_pred - alpha_target) / alpha_target) ** 2
        + 0.2 * ((g_pred - g_target) / g_target) ** 2
    )
    
    # 2. Selection Barrier Penalties
    min_t1 = selection.get("min_T1_us", 100.0)
    pen_t1 = max(0.0, (min_t1 - t1_pred) / min_t1) ** 2
    
    min_ratio = selection.get("min_ratio", 50.0)
    max_ratio = selection.get("max_ratio", 100.0)
    pen_ratio_low = max(0.0, (min_ratio - ratio_pred) / min_ratio) ** 2
    pen_ratio_high = max(0.0, (ratio_pred - max_ratio) / max_ratio) ** 2
    
    min_chi = selection.get("min_chi_MHz", 0.5)
    max_chi = selection.get("max_chi_MHz", 10.0)
    abs_chi = abs(chi_pred)
    pen_chi = max(0.0, (min_chi - abs_chi) / min_chi) ** 2 + max(0.0, (abs_chi - max_chi) / max_chi) ** 2
    
    max_disp = selection.get("max_dispersion_kHz", 10.0)
    pen_disp = max(0.0, (disp_pred - max_disp) / max_disp) ** 2
    
    # Penalty weights
    lambda_t1 = 800.0
    lambda_ratio = 1000.0
    lambda_chi = 400.0
    lambda_disp = 500.0
    
    total_cost = (
        loss_target
        + (lambda_t1 * pen_t1)
        + (lambda_ratio * (pen_ratio_low + pen_ratio_high))
        + (lambda_chi * pen_chi)
        + (lambda_disp * pen_disp)
    )
    return float(total_cost)

