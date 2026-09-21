import math
from qiskit_metal import view

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None

def to_meters(value):
    """
    Convert a dimensional value to meters.

    Examples
    --------
    10e-6
    "10um"
    "10 µm"
    "0.01mm"
    "10000nm"
    """

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = (
        str(value)
        .strip()
        .lower()
        .replace("µ", "u")
        .replace(" ", "")
    )

    if text.endswith("um"):
        return float(text[:-2]) * 1e-6

    if text.endswith("mm"):
        return float(text[:-2]) * 1e-3

    if text.endswith("nm"):
        return float(text[:-2]) * 1e-9

    if text.endswith("m"):
        return float(text[:-1])

    return float(text)

def brent_minimize(
    function,
    lower,
    upper,
    tolerance=1e-9,
    max_iterations=100,
):
    """
    Bounded derivative-free Brent minimization.

    Combines:
        - Successive parabolic interpolation
        - Golden-section fallback

    Palace is NOT called inside this function.
    """

    a = float(lower)
    b = float(upper)

    if a >= b:
        raise ValueError(
            "Lower bound must be smaller than upper bound."
        )

    golden = (
        3.0 - math.sqrt(5.0)
    ) / 2.0

    tiny = 1e-15

    x = a + golden * (b - a)

    w = x
    v = x

    fx = float(function(x))
    fw = fx
    fv = fx

    d = 0.0
    e = 0.0

    converged = False

    for iteration in range(
        1,
        max_iterations + 1,
    ):

        midpoint = 0.5 * (a + b)

        tol1 = (
            tolerance
            + tiny * abs(x)
        )

        tol2 = 2.0 * tol1

        # ----------------------------------------------------
        # CONVERGENCE
        # ----------------------------------------------------

        if abs(x - midpoint) <= (
            tol2
            - 0.5 * (b - a)
        ):

            converged = True
            break

        previous_e = e

        parabolic_accepted = False

        # ----------------------------------------------------
        # SUCCESSIVE PARABOLIC INTERPOLATION
        # ----------------------------------------------------

        if abs(e) > tol1:

            r = (
                (x - w)
                * (fx - fv)
            )

            q = (
                (x - v)
                * (fx - fw)
            )

            p = (
                (x - v) * q
                - (x - w) * r
            )

            q = 2.0 * (q - r)

            if q > 0:
                p = -p
            else:
                q = -q

            e = d

            if (
                q > 0
                and abs(p)
                < abs(
                    0.5
                    * q
                    * previous_e
                )
                and p > q * (a - x)
                and p < q * (b - x)
            ):

                d = p / q

                candidate = x + d

                if (
                    candidate - a < tol2
                    or
                    b - candidate < tol2
                ):

                    direction = midpoint - x

                    if direction == 0:
                        direction = 1.0

                    d = math.copysign(
                        tol1,
                        direction,
                    )

                parabolic_accepted = True

        # ----------------------------------------------------
        # GOLDEN-SECTION FALLBACK
        # ----------------------------------------------------

        if not parabolic_accepted:

            if x < midpoint:
                e = b - x
            else:
                e = a - x

            d = golden * e

        # ----------------------------------------------------
        # MINIMUM STEP
        # ----------------------------------------------------

        if abs(d) >= tol1:

            u = x + d

        else:

            direction = (
                d
                if d != 0
                else midpoint - x
            )

            if direction == 0:
                direction = 1.0

            u = (
                x
                + math.copysign(
                    tol1,
                    direction,
                )
            )

        u = min(
            max(u, a),
            b,
        )

        fu = float(
            function(u)
        )

        # ----------------------------------------------------
        # UPDATE BRENT STATE
        # ----------------------------------------------------

        if fu <= fx:

            if u >= x:
                a = x
            else:
                b = x

            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu

        else:

            if u < x:
                a = u
            else:
                b = u

            if (
                fu <= fw
                or w == x
            ):

                v, fv = w, fw
                w, fw = u, fu

            elif (
                fu <= fv
                or v == x
                or v == w
            ):

                v, fv = u, fu

    else:
        iteration = max_iterations

    return (
        x,
        fx,
        iteration,
        converged,
    )


# ============================================================
# BUILD PCHIP SURROGATE
# ============================================================

def build_surrogate(
    widths,
    frequencies,
):
    """
    Build a shape-preserving PCHIP surrogate.

    PCHIP is used instead of a global quadratic because it
    avoids artificial polynomial overshoot and nonsense
    extrapolation.

    Width:
        meters -> internally converted to µm

    Frequency:
        Hz -> internally converted to GHz
    """

    if PchipInterpolator is None:

        raise ImportError(
            "SciPy is required for PCHIP.\n"
            "Install it using:\n"
            "pip install scipy"
        )

    x = (
        np.asarray(
            widths,
            dtype=float,
        )
        * 1e6
    )

    y = (
        np.asarray(
            frequencies,
            dtype=float,
        )
        / 1e9
    )

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    # --------------------------------------------------------
    # REMOVE DUPLICATE WIDTHS
    # --------------------------------------------------------

    unique_x = []
    unique_y = []

    for xi, yi in zip(x, y):

        if (
            len(unique_x) == 0
            or abs(
                xi - unique_x[-1]
            ) > 1e-12
        ):

            unique_x.append(
                float(xi)
            )

            unique_y.append(
                float(yi)
            )

        else:

            # Keep latest result at duplicate width.
            unique_y[-1] = float(yi)

    if len(unique_x) < 2:

        raise RuntimeError(
            "At least two unique Palace samples "
            "are required to build the surrogate."
        )

    interpolator = PchipInterpolator(
        unique_x,
        unique_y,
        extrapolate=False,
    )

    minimum_width = (
        min(unique_x)
        * 1e-6
    )

    maximum_width = (
        max(unique_x)
        * 1e-6
    )

    def surrogate_frequency(width):

        width = float(width)

        if (
            width < minimum_width
            or width > maximum_width
        ):

            return float("nan")

        width_um = (
            width * 1e6
        )

        value = interpolator(
            width_um
        )

        value = float(value)

        if not math.isfinite(value):

            return float("nan")

        return value * 1e9

    return (
        surrogate_frequency,
        minimum_width,
        maximum_width,
    )


# ============================================================
# MAIN OPTIMIZER
# ============================================================

def optimize_resonator_width(
    target_frequency,
    initial_width,
    design,
    simulation,
    lower_bound,
    upper_bound,
    mode_index=0,
    frequency_tolerance=1e6,
    width_tolerance=1e-9,
    max_palace_calls=6,
    max_brent_iterations=100,
    minimum_frequency_variation=1e3,
):
    """
    Fixed-mesh surrogate-assisted Brent optimizer.

    NO adaptive meshing.
    NO fidelity switching.
    NO repeated final validation.

    Workflow
    --------

        3 Palace samples
              |
              v
        PCHIP surrogate
              |
              v
        Brent minimization
        (cheap, no Palace)
              |
              v
        Palace validation
              |
              v
        Add real sample
              |
              v
        Rebuild PCHIP
              |
              v
        Repeat until:
            - frequency tolerance reached, or
            - Palace budget exhausted

    The returned final point is always an ACTUAL Palace sample.
    """

    target_frequency = float(
        target_frequency
    )

    initial_width = to_meters(
        initial_width
    )

    lower_bound = to_meters(
        lower_bound
    )

    upper_bound = to_meters(
        upper_bound
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if lower_bound >= upper_bound:

        raise ValueError(
            "lower_bound must be smaller "
            "than upper_bound."
        )

    if not (
        lower_bound
        <= initial_width
        <= upper_bound
    ):

        raise ValueError(
            "initial_width must lie inside "
            "the search bounds."
        )

    if max_palace_calls < 4:

        raise ValueError(
            "max_palace_calls must be at least 4."
        )

    # ========================================================
    # STORAGE
    # ========================================================

    sampled_widths = []
    sampled_frequencies = []

    prediction_history = []

    palace_calls = 0
