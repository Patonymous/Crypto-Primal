"""
Atak Primal na FrodoKEM.

Ten moduł implementuje atak primal (Kannan's Embedding) na kryptosystem FrodoKEM.
FrodoKEM opiera się na problemie LWE z macierzami, więc atak jest analogiczny
do ataku na standardowe LWE, ale musi być przeprowadzony dla każdej kolumny
sekretu osobno (lub dla spłaszczonego problemu).

Strategia ataku:
1. Wydobyć instancję LWE z klucza publicznego FrodoKEM
2. Zbudować kratę Kannan's Embedding
3. Użyć redukcji LLL/BKZ do znalezienia krótkiego wektora
4. Wyodrębnić sekret z krótkiego wektora
5. Powtórzyć dla każdej kolumny sekretu
"""

import numpy as np
import time
from fpylll import IntegerMatrix, LLL, BKZ
from src.printer import Printer, Style
from src.frodo import FrodoKEM, center_modulo


def build_primal_lattice_frodo(A: np.ndarray, b: np.ndarray, q: int) -> IntegerMatrix:
    """
    Konstruuje macierz bazy dla ataku Primal (Kannan's Embedding) na FrodoKEM.

    Dla instancji LWE: b = As + e mod q (gdzie A jest m x n)
    
    Konstrukcja identyczna jak w oryginalnym ataku:
    
    B = [ qI_m   0     0 ]
        [  A^T   I_n   0 ]
        [  b^T   0     1 ]
    
    Dla FrodoKEM: A jest n x n, więc m = n.
    Wymiar kraty: (m + n + 1) = (2n + 1)
    
    Args:
        A: Macierz publiczna (m x n), dla FrodoKEM m = n
        b: Wektor publiczny (m,) - jedna kolumna B_pub
        q: Moduł
        
    Returns:
        Macierz bazy kraty w formacie fpylll
    """
    m, n = A.shape  # Dla FrodoKEM m = n
    d = m + n + 1   # Wymiar kraty
    
    # Tworzymy macierz w fpylll
    B = IntegerMatrix(d, d)
    
    # Wypełnianie bloku q*I_m (lewy górny róg)
    for i in range(m):
        B[i, i] = q
    
    # Wypełnianie bloku A^T (lewy środek) i I_n (prawy środek)
    # A ma wymiary m x n, więc A^T ma n x m.
    for i in range(n):
        row_idx = m + i
        # Wstawianie wiersza z A^T (czyli kolumny z A)
        for j in range(m):
            B[row_idx, j] = int(A[j, i])  # A[j, i] to element A^T[i, j]
        # Wstawianie I_n (diagonalnie obok A^T)
        B[row_idx, row_idx] = 1
    
    # Wypełnianie wektora b (ostatni wiersz)
    row_last = m + n
    for j in range(m):
        B[row_last, j] = int(b[j])
    
    # Wstawianie 1 na końcu (embedding factor)
    B[row_last, m + n] = 1
    
    return B


def check_lwe_solution(
    A: np.ndarray, b: np.ndarray, 
    candidate_e: np.ndarray, candidate_s: np.ndarray, 
    q: int, sigma: float, margin: float = 2.0
) -> bool:
    """
    Weryfikuje, czy (s, e) jest poprawnym rozwiązaniem instancji LWE.
    Sprawdza czy As jest blisko b (różnica powinna być małym błędem).
    
    Ta weryfikacja jest taka sama jak w oryginalnym ataku na LWE.
    """
    m = len(b)
    
    # Sprawdź czy As jest blisko b (różnica powinna być małym błędem)
    b_calc = A @ candidate_s % q
    diff = center_modulo(b - b_calc, q)
    
    # Jeśli norma różnicy jest mała, rozwiązanie jest poprawne
    norm_diff = np.linalg.norm(diff)
    expected_norm = np.sqrt(m) * sigma
    
    # Margines błędu dla weryfikacji - domyślnie 2x oczekiwanej normy
    return norm_diff < (expected_norm * margin)


def primal_attack_frodo_column(
    A: np.ndarray, b: np.ndarray, q: int, sigma: float, 
    block_size: int = 20
) -> tuple[np.ndarray | None, float]:
    """
    Przeprowadza atak Primal na pojedynczą kolumnę FrodoKEM (instancję LWE).
    
    Args:
        A: Macierz publiczna (m x n), dla FrodoKEM m = n
        b: Wektor publiczny (m,) - jedna kolumna B_pub
        q: Moduł
        sigma: Odchylenie standardowe błędu
        block_size: Rozmiar bloku dla BKZ
        
    Returns:
        Tuple (odzyskany_sekret, czas_redukcji)
    """
    m, n = A.shape
    
    with Printer(Style.LOG) as printer:
        printer(f"  Wymiar kraty: {m + n + 1} x {m + n + 1}")
    
    start_time = time.time()
    
    # 1. Buduj kratę
    with Printer(Style.LOG) as printer:
        printer(f"  Konstruowanie kraty...", end="")
    B = build_primal_lattice_frodo(A, b, q)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    # 2. Redukcja LLL (wstępna)
    with Printer(Style.LOG) as printer:
        printer(f"  Redukcja LLL...", end="")
    LLL.reduction(B)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    # 3. Redukcja BKZ (silniejsza)
    with Printer(Style.LOG) as printer:
        printer(f"  Redukcja BKZ-{block_size}...", end="")
    BKZ.reduction(B, BKZ.Param(block_size))
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    reduction_time = time.time() - start_time
    
    # 4. Szukaj sekretu w zredukowanej bazie
    # Spodziewamy się wektora v = (e, s, 1) lub v = (-e, -s, -1)
    # Sekret s znajduje się na indeksach [m : m+n]
    with Printer(Style.LOG) as printer:
        printer(f"  Poszukiwanie sekretu...", end="")
    
    found_s = None
    for row_idx in range(B.nrows):
        v = B[row_idx]
        
        # Pobierz potencjalny sekret (kandydat)
        embedding_factor = v[m + n]
        e_scaled = np.array([v[j] for j in range(0, m)])
        s_scaled = np.array([v[j] for j in range(m, m + n)])
        
        # Oblicz odwrotność modularną współczynnika embedding
        try:
            factor_inv = pow(int(embedding_factor), -1, int(q))
        except ValueError:
            # Nie istnieje odwrotność modularna (gcd(k, q) != 1)
            continue
        
        # Pomnóż przez odwrotność modularną aby "podzielić" w arytmetyce modularnej
        e_candidate = (e_scaled * factor_inv) % q
        s_candidate = (s_scaled * factor_inv) % q
        
        if check_lwe_solution(A, b, e_candidate, s_candidate, q, sigma):
            found_s = s_candidate
            break
        
        e_candidate = (-e_candidate) % q
        s_candidate = (-s_candidate) % q
        
        if check_lwe_solution(A, b, e_candidate, s_candidate, q, sigma):
            found_s = s_candidate
            break
    
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    return found_s, reduction_time


def primal_attack_frodo(
    frodo: FrodoKEM, block_size: int = 20, max_columns: int | None = None
) -> tuple[np.ndarray | None, float]:
    """
    Przeprowadza pełny atak Primal na FrodoKEM.
    
    Atak polega na odzyskaniu sekretu S kolumna po kolumnie.
    Dla każdej kolumny B_pub[:, i] mamy:
        B_pub[:, i] = A @ S[:, i] + E[:, i] mod q
    
    Jest to standardowa instancja LWE, którą możemy atakować osobno.
    
    Args:
        frodo: Instancja FrodoKEM z wygenerowanymi kluczami
        block_size: Rozmiar bloku dla BKZ
        max_columns: Maksymalna liczba kolumn do atakowania (None = wszystkie)
        
    Returns:
        Tuple (odzyskana_macierz_sekretu, całkowity_czas)
    """
    if frodo.A is None or frodo.B_pub is None:
        raise ValueError("FrodoKEM nie ma wygenerowanych kluczy publicznych.")
    
    n = frodo.params.n
    m = frodo.params.m
    n_bar = frodo.params.n_bar
    q = frodo.params.q
    sigma = frodo.params.sigma
    
    # Określ ile kolumn atakować
    columns_to_attack = n_bar if max_columns is None else min(max_columns, n_bar)
    
    with Printer(Style.INFO) as printer:
        printer(f"Rozpoczynam atak Primal na FrodoKEM")
        printer(f"Parametry: n={n}, m={m}, n_bar={n_bar}, q={q}, sigma={sigma:.2f}")
        printer(f"Kolumn do atakowania: {columns_to_attack}")
    
    total_start = time.time()
    
    # Odzyskaj każdą kolumnę sekretu
    recovered_S = np.zeros((n, columns_to_attack), dtype=int)
    success_count = 0
    
    for col_idx in range(columns_to_attack):
        with Printer(Style.INFO) as printer:
            printer(f"\nAtakuję kolumnę {col_idx + 1}/{columns_to_attack}:")
        
        # Wyodrębnij instancję LWE dla tej kolumny
        A = frodo.A
        b = frodo.B_pub[:, col_idx]
        
        # Przeprowadź atak
        recovered_s, col_time = primal_attack_frodo_column(A, b, q, sigma, block_size)
        
        if recovered_s is not None:
            recovered_S[:, col_idx] = recovered_s
            success_count += 1
            
            with Printer(Style.SUCCESS) as printer:
                printer(f"  Kolumna {col_idx + 1}: SUKCES (czas: {col_time:.2f}s)")
            
            # Weryfikuj z prawdziwym sekretem (jeśli dostępny)
            if frodo._S is not None:
                is_correct = frodo.check_column_recovery(col_idx, recovered_s)
                if is_correct:
                    with Printer(Style.SUCCESS) as printer:
                        printer(f"  Weryfikacja: sekret POPRAWNY")
                else:
                    with Printer(Style.FAIL) as printer:
                        printer(f"  Weryfikacja: sekret NIEPOPRAWNY")
        else:
            with Printer(Style.FAIL) as printer:
                printer(f"  Kolumna {col_idx + 1}: PORAŻKA (czas: {col_time:.2f}s)")
    
    total_time = time.time() - total_start
    
    with Printer(Style.INFO) as printer:
        printer(f"\nPodsumowanie ataku:")
        printer(f"  Odzyskano: {success_count}/{columns_to_attack} kolumn")
        printer(f"  Całkowity czas: {total_time:.2f}s")
    
    if success_count == columns_to_attack:
        return recovered_S, total_time
    else:
        return None, total_time


def demo_key_recovery_attack(frodo: FrodoKEM, block_size: int = 20) -> bool:
    """
    Demonstracja ataku z odzyskaniem klucza i próbą dekapsulacji.
    
    1. Atakuje FrodoKEM i odzyskuje sekret S
    2. Używa odzyskanego sekretu do dekapsulacji
    3. Porównuje z prawidłowym kluczem współdzielonym
    
    Returns:
        True jeśli atak się powiódł (odzyskano poprawny klucz)
    """
    with Printer(Style.INFO) as printer:
        printer("=" * 50)
        printer("DEMONSTRACJA ATAKU NA FRODOKEM")
        printer("=" * 50)
    
    # Generuj szyfrogram (symulacja enkapsulacji przez kogoś innego)
    with Printer(Style.LOG) as printer:
        printer("\nGenerowanie szyfrogramu (enkapsulacja)...", end="")
    ciphertext, true_shared_secret = frodo.encaps()
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    with Printer(Style.INFO) as printer:
        printer(f"Prawdziwy klucz współdzielony: {true_shared_secret[:16].hex()}...")
    
    # Przeprowadź atak
    recovered_S, attack_time = primal_attack_frodo(frodo, block_size)
    
    if recovered_S is None:
        with Printer(Style.FAIL) as printer:
            printer("\nAtak nie powiódł się - nie udało się odzyskać sekretu.")
        return False
    
    # Utwórz kopię FrodoKEM z odzyskanym sekretem
    with Printer(Style.LOG) as printer:
        printer("\nPróba dekapsulacji z odzyskanym sekretem...", end="")
    
    # Zapisz oryginalny sekret i podstaw odzyskany
    original_S = frodo._S
    frodo._S = recovered_S
    
    try:
        recovered_shared_secret = frodo.decaps(ciphertext)
        with Printer(Style.LOG) as printer:
            printer(" Wykonano")
        
        with Printer(Style.INFO) as printer:
            printer(f"Odzyskany klucz współdzielony: {recovered_shared_secret[:16].hex()}...")
        
        # Porównaj klucze
        if true_shared_secret == recovered_shared_secret:
            with Printer(Style.SUCCESS) as printer:
                printer("\n" + "=" * 50)
                printer("ATAK ZAKOŃCZONY SUKCESEM!")
                printer("Odzyskano poprawny klucz współdzielony!")
                printer("=" * 50)
            return True
        else:
            with Printer(Style.FAIL) as printer:
                printer("\nKlucze nie są zgodne - dekapsulacja nieudana.")
            return False
    finally:
        # Przywróć oryginalny sekret
        frodo._S = original_S
