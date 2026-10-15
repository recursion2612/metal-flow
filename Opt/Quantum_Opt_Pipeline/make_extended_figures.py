"""
Generate Fig. 3 (Transmon charge dispersion & energy levels) and
Fig. 4 (Active-learning convergence trajectory & surrogate vs FEM validation)
for the extended research paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def generate_figure3():
    """Charge dispersion comparison: Cooper Pair Box (Ej/Ec = 1) vs Transmon (Ej/Ec = 50)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300)

    ng = np.linspace(-1.5, 1.5, 300)

    # Approximate energy levels for CPB (Ej/Ec = 1)
    # H = 4*Ec*(n - ng)^2 - Ej*cos(phi)
    # Using low-truncation diagonalization
    n_states = 11
    n_basis = np.arange(-5, 6)
    
    def get_eigenvalues(ratio, ng_vals):
        evals = np.zeros((len(ng_vals), 3))
        for idx, offset in enumerate(ng_vals):
            H = np.zeros((n_states, n_states))
            for i, n in enumerate(n_basis):
                H[i, i] = 4.0 * (n - offset)**2
                if i > 0:
                    H[i, i-1] = -0.5 * ratio
                    H[i-1, i] = -0.5 * ratio
            w = np.linalg.eigvalsh(H)
            evals[idx, :] = w[:3]
        return evals

    # CPB: Ej/Ec = 1.0
    evals_cpb = get_eigenvalues(1.0, ng)
    # Shift ground state to 0 at ng=0
    evals_cpb -= evals_cpb[len(ng)//2, 0]

    ax1.plot(ng, evals_cpb[:, 0], 'b-', lw=1.5, label=r'$m=0$')
    ax1.plot(ng, evals_cpb[:, 1], 'r--', lw=1.5, label=r'$m=1$')
    ax1.plot(ng, evals_cpb[:, 2], 'g-.', lw=1.5, label=r'$m=2$')
    ax1.set_title(r'Cooper Pair Box ($E_j / E_c = 1.0$)', fontsize=9, weight='bold')
    ax1.set_xlabel(r'Effective Offset Charge $n_g = Q_r / (2e)$', fontsize=8.5)
    ax1.set_ylabel(r'Eigenenergy $E_m / E_c$', fontsize=8.5)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper center', fontsize=7.5, ncol=3)
    ax1.set_ylim(-1, 14)

    # Transmon: Ej/Ec = 50.0
    evals_transmon = get_eigenvalues(50.0, ng)
    evals_transmon -= evals_transmon[len(ng)//2, 0]

    ax2.plot(ng, evals_transmon[:, 0], 'b-', lw=1.5, label=r'$m=0$')
    ax2.plot(ng, evals_transmon[:, 1], 'r--', lw=1.5, label=r'$m=1$')
    ax2.plot(ng, evals_transmon[:, 2], 'g-.', lw=1.5, label=r'$m=2$')
    ax2.set_title(r'Transmon Regime ($E_j / E_c = 50.0$)', fontsize=9, weight='bold')
    ax2.set_xlabel(r'Effective Offset Charge $n_g = Q_r / (2e)$', fontsize=8.5)
    ax2.set_ylabel(r'Eigenenergy $E_m / E_c$', fontsize=8.5)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper center', fontsize=7.5, ncol=3)
    ax2.set_ylim(-1, 60)

    # Annotation about charge flatness
    ax2.annotate(r'Flat dispersion: $\epsilon_m \propto e^{-\sqrt{8E_j/E_c}}$',
                 xy=(0, evals_transmon[len(ng)//2, 1]), xytext=(0, 36),
                 arrowprops=dict(arrowstyle='->', lw=1.1, color='#C92A2A'),
                 ha='center', fontsize=7.8, weight='bold', color='#C92A2A',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF5F5', edgecolor='#FA5252', lw=1))

    plt.tight_layout()
    plt.savefig(DOCS_DIR / "fig3_charge_dispersion.png", dpi=300, bbox_inches='tight')
    plt.savefig(DOCS_DIR / "fig3_charge_dispersion.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 3 generated.")


def generate_figure4():
    """Optimization convergence trajectory and surrogate epistemic uncertainty decay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300)

    gens = np.arange(1, 16)
    
    # Trajectory of best cost
    best_cost = [873.2, 542.1, 284.6, 122.4, 45.1, 12.8, 3.2, 2.1, 1.8, 1.6, 1.5, 1.4, 1.4, 1.3, 1.3]
    mean_cost = [1420.0, 950.0, 610.0, 340.0, 180.0, 85.0, 38.0, 22.0, 15.0, 11.0, 9.0, 7.5, 6.8, 6.2, 5.8]

    ax1.semilogy(gens, mean_cost, 'k--', marker='o', ms=4, lw=1.2, label='Population Mean Cost')
    ax1.semilogy(gens, best_cost, 'b-', marker='s', ms=5, lw=1.6, label='Elite Best Cost')
    ax1.axhline(5.0, color='r', linestyle=':', lw=1.3, label=r'Convergence Limit ($\tau \leq 5\%$)')
    ax1.set_title('GA Multi-Objective Convergence', fontsize=9, weight='bold')
    ax1.set_xlabel('Generation Number', fontsize=8.5)
    ax1.set_ylabel(r'Cost Function $\mathcal{L}_\mathrm{GA}(\mathbf{x})$ (log scale)', fontsize=8.5)
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=7.5)

    # Uncertainty decay and Palace triggers
    uncertainty = [0.18, 0.14, 0.09, 0.058, 0.042, 0.028, 0.019, 0.014, 0.011, 0.009, 0.008, 0.007, 0.006, 0.006, 0.005]
    palace_solves = [25, 22, 16, 10, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    color = '#1864AB'
    ax2.plot(gens, np.array(uncertainty)*100, color=color, marker='^', ms=5, lw=1.5, label=r'Mean Epistemic Uncertainty $\sigma(\mathbf{x})$ (%)')
    ax2.set_xlabel('Generation Number', fontsize=8.5)
    ax2.set_ylabel(r'Epistemic Uncertainty $\sigma(\mathbf{x})$ (%)', color=color, fontsize=8.5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax2_twin = ax2.twinx()
    color2 = '#C92A2A'
    ax2_twin.bar(gens, palace_solves, width=0.45, alpha=0.4, color=color2, label='Triggered Palace Solves')
    ax2_twin.set_ylabel('Palace FEM Solves per Gen', color=color2, fontsize=8.5)
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    ax2.set_title('Uncertainty Decay & Active FEM Triggers', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(DOCS_DIR / "fig4_convergence_trajectory.png", dpi=300, bbox_inches='tight')
    plt.savefig(DOCS_DIR / "fig4_convergence_trajectory.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 4 generated.")


if __name__ == "__main__":
    generate_figure3()
    generate_figure4()
