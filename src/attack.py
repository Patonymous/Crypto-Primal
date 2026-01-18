import numpy as np
import time
from fpylll import IntegerMatrix, LLL, BKZ
from src.printer import Printer, Style
from src.lwe import LWE


def build_primal_lattice(lwe: LWE) -> IntegerMatrix:
    """
    Konstruuje macierz bazy dla ataku Primal (Kannan's Embedding).

    B = [ qI_m   0     0 ]
        [  A^T   I_n   0 ]
        [  b^T   0     1 ]
    """
    m, n = lwe.A.shape
    d = m + n + 1

    # Tworzymy macierz w fpylll (IntegerMatrix)
    B = IntegerMatrix(d, d)

    # Wypełnianie bloku q*I_m (lewy górny róg)
    for i in range(m):
        B[i, i] = lwe.q

    # Wypełnianie bloku A^T (lewy środek) i I_n (prawy środek)
    # A ma wymiary m x n, więc A^T ma n x m.
    # W fpylll indeksujemy B[wiersz, kolumna]
    for i in range(n):
        row_idx = m + i
        # Wstawianie wiersza z A^T (czyli kolumny z A)
        for j in range(m):
            B[row_idx, j] = int(lwe.A[j, i])  # A[j, i] to element A^T[i, j]

        # Wstawianie I_n (diagonalnie obok A^T)
        B[row_idx, row_idx] = 1

    # Wypełnianie wektora b (ostatni wiersz)
    row_last = m + n
    for j in range(m):
        B[row_last, j] = int(lwe.b[j])

    # Wstawianie 1 na końcu (embedding factor)
    B[row_last, m + n] = 1

    return B


def primal_attack(lwe: LWE, block_size: int) -> tuple[np.ndarray | None, float]:
    """
    Przeprowadza atak Primal używając redukcji LLL i BKZ.
    """
    m, n = lwe.A.shape
    with Printer(Style.LOG) as printer:
        printer(f"Wymiar kraty: {m + n + 1} x {m + n + 1}")

    start_time = time.time()

    # 1. Konstrukcja kraty
    with Printer(Style.LOG) as printer:
        printer(f"Konstruowanie kraty...", end="")
    B = build_primal_lattice(lwe)
    with Printer(Style.LOG) as printer:
        printer(f" Wykonano")

    # 2. Redukcja LLL (wstępna)
    with Printer(Style.LOG) as printer:
        printer(f"Redukcja LLL...", end="")
    LLL.reduction(B)
    with Printer(Style.LOG) as printer:
        printer(f" Wykonano")

    # 3. Redukcja BKZ (silniejsza)
    with Printer(Style.LOG) as printer:
        printer(f"Redukcja BKZ-{block_size}...", end="")
    BKZ.reduction(B, BKZ.Param(block_size))
    with Printer(Style.LOG) as printer:
        printer(f" Wykonano")

    reduction_time = time.time() - start_time

    # 4. Poszukiwanie sekretu w zredukowanej bazie
    # Spodziewamy się wektora v = (e, s, 1) lub v = (-e, -s, -1)
    # Sekret s znajduje się na indeksach [m : m+n]

    with Printer(Style.LOG) as printer:
        printer(f"Poszukiwanie sekretu w zredukowanej bazie...", end="")

    # Sprawdzamy wektory aż któryś się nada
    found_s = None
    for row_idx in range(B.nrows):
        v = B[row_idx]

        # Pobierz potencjalny sekret (kandydat)
        # W fpylll wektory są indeksowane jak listy
        embedding_factor = v[m + n]
        e_scaled = np.array([v[j] for j in range(0, m)])
        s_scaled = np.array([v[j] for j in range(m, m + n)])

        # Oblicz odwrotność modularną współczynnika homograficznego
        try:
            factor_inv = pow(int(embedding_factor), -1, int(lwe.q))
        except ValueError:
            # Nie istnieje odwrotność modularna (gcd(k, q) != 1)
            # Nie powinno się zdarzyć, gdy q jest pierwsze
            continue

        # Pomnóż przez odwrotność modularną aby "podzielić" w arytmetyce modularnej
        e_candidate = (e_scaled * factor_inv) % lwe.q
        s_candidate = (s_scaled * factor_inv) % lwe.q

        if lwe.check_solution_correct(e_candidate, s_candidate):
            found_s = s_candidate
            break

        e_candidate = (-e_candidate) % lwe.q
        s_candidate = (-s_candidate) % lwe.q

        if lwe.check_solution_correct(e_candidate, s_candidate):
            found_s = s_candidate
            break

    with Printer(Style.LOG) as printer:
        printer(f" Wykonano")

    return found_s, reduction_time
