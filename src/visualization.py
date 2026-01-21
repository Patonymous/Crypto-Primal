"""
Moduł wizualizacji ataku Primal na LWE.

Generuje wykresy pokazujące:
- Jakość redukcji kraty (normy wektorów)
- Profil bazy Gram-Schmidta
- Wpływ rozmiaru bloku BKZ
"""

import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, GSO
from pathlib import Path
from src.lwe import LWE
from src.attack import build_primal_lattice

# Opcjonalny import matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


def _check_matplotlib():
    """Sprawdza czy matplotlib jest dostępny."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib nie jest zainstalowany.\n"
            "Zainstaluj go komendą: pip install matplotlib"
        )


def _is_interactive():
    """Sprawdza czy backend matplotlib jest interaktywny."""
    if not MATPLOTLIB_AVAILABLE:
        return False
    import matplotlib
    return matplotlib.is_interactive() or matplotlib.get_backend() not in ['agg', 'Agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg']


def compute_basis_norms(B: IntegerMatrix) -> list[float]:
    """Oblicza normy wektorów w bazie."""
    return [np.linalg.norm([B[i, j] for j in range(B.ncols)]) for i in range(B.nrows)]


def compute_gs_norms(B: IntegerMatrix) -> list[float]:
    """Oblicza normy wektorów Gram-Schmidta."""
    M = GSO.Mat(B)
    M.update_gso()
    return [M.get_r(i, i) ** 0.5 for i in range(B.nrows)]


def visualize_reduction(lwe: LWE, block_sizes: list[int] = None, save_path: str = None):
    """
    Wizualizuje jakość redukcji dla różnych algorytmów.
    
    Args:
        lwe: Instancja LWE
        block_sizes: Lista rozmiarów bloków BKZ (domyślnie [10, 20, 30])
        save_path: Ścieżka do zapisu wykresu
    """
    _check_matplotlib()
    
    if block_sizes is None:
        block_sizes = [10, 20, 30]
    
    B_orig = build_primal_lattice(lwe)
    norms_orig = compute_basis_norms(B_orig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['b', 'g', 'r', 'm', 'c']
    
    # Wykres 1: Normy wektorów bazy
    ax1.semilogy(norms_orig, 'k--', label='Oryginalna', linewidth=2, alpha=0.5)
    
    B_lll = B_orig.__copy__()
    LLL.reduction(B_lll)
    ax1.semilogy(compute_basis_norms(B_lll), 'b-', label='LLL', linewidth=2)
    
    for i, bs in enumerate(block_sizes):
        B_bkz = B_lll.__copy__()
        BKZ.reduction(B_bkz, BKZ.Param(bs))
        ax1.semilogy(compute_basis_norms(B_bkz), f'{colors[(i+1) % len(colors)]}-', 
                    label=f'BKZ-{bs}', linewidth=2)
    
    ax1.set_xlabel('Indeks wektora')
    ax1.set_ylabel('Norma (log)')
    ax1.set_title('Normy wektorów bazy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Wykres 2: Profil Gram-Schmidta
    ax2.semilogy(compute_gs_norms(B_orig), 'k--', label='Oryginalna', linewidth=2, alpha=0.5)
    ax2.semilogy(compute_gs_norms(B_lll), 'b-', label='LLL', linewidth=2)
    
    for i, bs in enumerate(block_sizes):
        B_bkz = B_lll.__copy__()
        BKZ.reduction(B_bkz, BKZ.Param(bs))
        ax2.semilogy(compute_gs_norms(B_bkz), f'{colors[(i+1) % len(colors)]}-', 
                    label=f'BKZ-{bs}', linewidth=2)
    
    ax2.set_xlabel('Indeks wektora')
    ax2.set_ylabel('Norma GS (log)')
    ax2.set_title('Profil Gram-Schmidta')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano: {save_path}")
    
    if _is_interactive():
        plt.show()
    return fig


def visualize_block_size_impact(lwe: LWE, max_block: int = 40, save_path: str = None):
    """
    Pokazuje jak norma pierwszego wektora zmienia się z rozmiarem bloku BKZ.
    """
    _check_matplotlib()
    
    B = build_primal_lattice(lwe)
    LLL.reduction(B)
    
    block_sizes = list(range(5, min(max_block, B.nrows // 2), 5))
    first_norms = []
    
    for bs in block_sizes:
        B_copy = B.__copy__()
        BKZ.reduction(B_copy, BKZ.Param(bs))
        first_norms.append(compute_basis_norms(B_copy)[0])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(block_sizes, first_norms, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Rozmiar bloku BKZ')
    ax.set_ylabel('Norma pierwszego wektora')
    ax.set_title('Wpływ rozmiaru bloku BKZ na jakość redukcji')
    ax.grid(True, alpha=0.3)
    
    # Zaznacz minimum
    min_idx = np.argmin(first_norms)
    ax.annotate(f'Min: {first_norms[min_idx]:.1f}\n(BKZ-{block_sizes[min_idx]})',
               xy=(block_sizes[min_idx], first_norms[min_idx]),
               xytext=(10, 20), textcoords='offset points',
               bbox=dict(boxstyle='round', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano: {save_path}")
    
    if _is_interactive():
        plt.show()
    return fig


def plot_success_heatmap(results: dict, save_path: str = None):
    """
    Tworzy heatmapę sukcesu ataku.
    
    Args:
        results: Słownik {(n, sigma): success_rate}
    """
    _check_matplotlib()
    n_vals = sorted(set(k[0] for k in results.keys()))
    sigma_vals = sorted(set(k[1] for k in results.keys()))
    
    matrix = np.zeros((len(sigma_vals), len(n_vals)))
    for i, s in enumerate(sigma_vals):
        for j, n in enumerate(n_vals):
            matrix[i, j] = results.get((n, s), 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(n_vals)))
    ax.set_yticks(range(len(sigma_vals)))
    ax.set_xticklabels(n_vals)
    ax.set_yticklabels([f'{s:.2f}' for s in sigma_vals])
    ax.set_xlabel('Wymiar n')
    ax.set_ylabel('Odchylenie σ')
    ax.set_title('Współczynnik sukcesu ataku Primal')
    
    for i in range(len(sigma_vals)):
        for j in range(len(n_vals)):
            ax.text(j, i, f'{matrix[i, j]:.0%}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Sukces')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano: {save_path}")
    
    if _is_interactive():
        plt.show()
    return fig
