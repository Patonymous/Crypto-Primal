"""
Tryb interaktywny ataku Primal.

Pozwala krok po kroku obserwować przebieg ataku:
- Budowa kraty
- Redukcja LLL
- Redukcja BKZ
- Szukanie sekretu

Idealny do prezentacji i nauki.
"""

import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, GSO
from src.lwe import LWE
from src.attack import build_primal_lattice
from src.printer import Printer, Style


def format_vector(v, max_elements: int = 10) -> str:
    """Formatuje wektor do wyświetlenia."""
    elements = [v[i] for i in range(min(len(v), max_elements))]
    if len(v) > max_elements:
        return f"[{', '.join(map(str, elements))}, ...]"
    return f"[{', '.join(map(str, elements))}]"


def compute_norm(v) -> float:
    """Oblicza normę wektora fpylll."""
    return np.linalg.norm([v[i] for i in range(len(v))])


def print_lattice_info(B: IntegerMatrix, title: str = "Stan kraty"):
    """Wyświetla informacje o kracie."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print('=' * 50)
    print(f"Wymiar: {B.nrows} x {B.ncols}")
    
    # Pierwsze 5 wektorów
    print("\nPierwsze wektory bazy:")
    for i in range(min(5, B.nrows)):
        v = B[i]
        norm = compute_norm(v)
        print(f"  v[{i}]: norma = {norm:.2f}")
        print(f"         {format_vector(v)}")
    
    # Statystyki norm
    norms = [compute_norm(B[i]) for i in range(B.nrows)]
    print(f"\nStatystyki norm:")
    print(f"  Min:    {min(norms):.2f}")
    print(f"  Max:    {max(norms):.2f}")
    print(f"  Średnia:{np.mean(norms):.2f}")


def wait_for_user(message: str = "Naciśnij Enter aby kontynuować..."):
    """Czeka na akcję użytkownika."""
    input(f"\n>>> {message}")


def interactive_attack(lwe: LWE, block_size: int = 20, auto: bool = False):
    """
    Przeprowadza atak w trybie interaktywnym.
    
    Args:
        lwe: Instancja LWE
        block_size: Rozmiar bloku BKZ
        auto: Jeśli True, nie czeka na użytkownika
    """
    m, n = lwe.m, lwe.n
    alpha = lwe.sigma / lwe.q  # Oblicz alpha z sigma
    
    print("\n" + "=" * 60)
    print("      INTERAKTYWNY ATAK PRIMAL NA LWE")
    print("=" * 60)
    print(f"\nParametry: n={n}, m={m}, q={lwe.q}, α={alpha:.4f}")
    print(f"Sekret s znany (do weryfikacji): {lwe._s[:5]}...")
    
    if not auto:
        wait_for_user("Naciśnij Enter aby rozpocząć budowę kraty...")
    
    # KROK 1: Budowa kraty
    print("\n" + "-" * 60)
    print("KROK 1: Budowa kraty Kannan Embedding")
    print("-" * 60)
    print("""
Konstrukcja macierzy bazy:

    B = [ qI_m   0     0 ]     Wymiar: (m+n+1) x (m+n+1)
        [  A^T   I_n   0 ]
        [  b^T   0     1 ]

Szukamy wektora (e, s, 1) który jest KRÓTKI
(bo e i s mają małe elementy z rozkładu błędów).
""")
    
    B = build_primal_lattice(lwe)
    print_lattice_info(B, "Oryginalna krata")
    
    if not auto:
        wait_for_user("Naciśnij Enter aby wykonać redukcję LLL...")
    
    # KROK 2: Redukcja LLL
    print("\n" + "-" * 60)
    print("KROK 2: Redukcja LLL")
    print("-" * 60)
    print("""
Algorytm Lenstra-Lenstra-Lovász:
- Czas: wielomianowy O(d^5 log^3 B)
- Jakość: słaba gwarancja (2^(d/2) * λ_1)
- Ale szybki i daje dobrą bazę startową dla BKZ
""")
    
    LLL.reduction(B)
    print_lattice_info(B, "Po redukcji LLL")
    
    # Sprawdź czy LLL wystarczy
    found_s = try_extract_secret(B, lwe)
    if found_s is not None:
        print("\n✓ Sekret znaleziony już po LLL!")
        print(f"  s = {found_s}")
        return found_s
    else:
        print("\n✗ LLL nie wystarczyło, potrzebna mocniejsza redukcja.")
    
    if not auto:
        wait_for_user(f"Naciśnij Enter aby wykonać redukcję BKZ-{block_size}...")
    
    # KROK 3: Redukcja BKZ
    print("\n" + "-" * 60)
    print(f"KROK 3: Redukcja BKZ-{block_size}")
    print("-" * 60)
    print(f"""
Block Korkine-Zolotarev (BKZ):
- Przetwarza bloki rozmiaru β = {block_size}
- W każdym bloku rozwiązuje SVP (Shortest Vector Problem)
- Większy blok = lepsza redukcja, ale wolniej
- Czas: wykładniczy w β, ale praktycznie wykonalny dla β < 60
""")
    
    BKZ.reduction(B, BKZ.Param(block_size))
    print_lattice_info(B, f"Po redukcji BKZ-{block_size}")
    
    if not auto:
        wait_for_user("Naciśnij Enter aby szukać sekretu...")
    
    # KROK 4: Ekstrakcja sekretu
    print("\n" + "-" * 60)
    print("KROK 4: Ekstrakcja sekretu")
    print("-" * 60)
    print("""
Szukamy wektora postaci (e, s, k) w zredukowanej bazie.
- Sprawdzamy każdy wektor bazy
- Dla k ≠ 0 obliczamy s = s_scaled / k mod q
- Weryfikujemy czy As + e ≡ b (mod q)
""")
    
    found_s = try_extract_secret(B, lwe, verbose=True)
    
    print("\n" + "=" * 60)
    if found_s is not None:
        print("      SUKCES! SEKRET ODZYSKANY!")
        print("=" * 60)
        print(f"Odzyskany s: {found_s}")
        print(f"Prawdziwy s: {lwe._s}")
        print(f"Zgodność:    {np.array_equal(found_s, lwe._s)}")
    else:
        print("      PORAŻKA - NIE ZNALEZIONO SEKRETU")
        print("=" * 60)
        print("Możliwe przyczyny:")
        print("  - Za mały rozmiar bloku BKZ")
        print("  - Zbyt trudne parametry LWE")
        print("  - Potrzeba więcej prób")
    
    return found_s


def try_extract_secret(B: IntegerMatrix, lwe: LWE, verbose: bool = False) -> np.ndarray:
    """Próbuje wyodrębnić sekret z zredukowanej kraty."""
    m, n = lwe.m, lwe.n
    
    for row_idx in range(B.nrows):
        v = B[row_idx]
        embedding_factor = v[m + n]
        
        if embedding_factor == 0:
            continue
        
        try:
            factor_inv = pow(int(embedding_factor), -1, int(lwe.q))
        except ValueError:
            continue
        
        e_scaled = np.array([v[j] for j in range(m)])
        s_scaled = np.array([v[j] for j in range(m, m + n)])
        
        # Sprawdź obie orientacje
        for sign in [1, -1]:
            e_candidate = (sign * e_scaled * factor_inv) % lwe.q
            s_candidate = (sign * s_scaled * factor_inv) % lwe.q
            
            if verbose and row_idx < 3:
                print(f"\nPróba v[{row_idx}], k={embedding_factor}, znak={sign}:")
                print(f"  s_kandydat = {format_vector(s_candidate)}")
            
            if lwe.check_solution_correct(e_candidate, s_candidate):
                if verbose:
                    print(f"  → ZNALEZIONO!")
                return s_candidate
    
    return None


def demo_interactive():
    """Demonstracja trybu interaktywnego."""
    print("\n" + "=" * 60)
    print("    DEMONSTRACJA TRYBU INTERAKTYWNEGO")
    print("=" * 60)
    
    # Łatwe parametry
    lwe = LWE(n=10, m=60, q=101, alpha=0.01)
    lwe.generate()
    
    interactive_attack(lwe, block_size=20, auto=True)


if __name__ == "__main__":
    demo_interactive()
