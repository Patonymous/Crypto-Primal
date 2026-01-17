#!/usr/bin/env python3

import numpy as np
import time
from fpylll import IntegerMatrix, LLL, BKZ


class LWEInstance:
    """
    Klasa pomocnicza przechowująca instancję problemu LWE.
    """

    def __init__(self, n: int, m: int, q: int, alpha: float):
        self.n = n  # Wymiar sekretu
        self.m = m  # Liczba próbek
        self.q = q  # Moduł
        self.alpha = alpha  # Odchylenie standardowe szumu

        self.A = None
        self.s = None
        self.e = None
        self.b = None

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Generuje losową instancję LWE: b = As + e mod q."""

        # 1. Macierz publiczna A (m x n)
        self.A = np.random.randint(0, self.q, size=(self.m, self.n))

        # 2. Sekret s (n) - losowy jednostajny
        self.s = np.random.randint(0, self.q, size=self.n)

        # 3. Błąd e (m) - rozkład Gaussa, zaokrąglony do liczb całkowitych
        sigma = self.alpha * self.q
        # W praktyce LWE często używa dyskretnego rozkładu Gaussa, tutaj przybliżenie:
        self.e = np.round(np.random.normal(0, sigma, size=self.m)).astype(int)

        self.b = (self.A @ self.s + self.e) % self.q

        return self.A, self.b

    def check_solution(self, candidate_s: np.ndarray | None) -> bool:
        """Weryfikuje, czy podany kandydat na sekret jest poprawny."""

        if candidate_s is None:
            return False

        if self.A is None or self.b is None:
            raise ValueError("Instancja LWE nie została wygenerowana.")

        # Sprawdź czy As' jest blisko b (różnica powinna być małym błędem)
        b_calc = self.A @ candidate_s % self.q
        diff = (
            self.b - b_calc + self.q // 2
        ) % self.q - self.q // 2  # Centrowanie modulo

        # Jeśli norma różnicy jest mała, rozwiązanie jest poprawne
        norm_diff = np.linalg.norm(diff)
        expected_norm = np.sqrt(self.m) * (self.alpha * self.q)

        # Margines błędu dla weryfikacji - 2x oczekiwanej normy
        is_correct = norm_diff < (expected_norm * 2.0)

        # Dodatkowa weryfikacja dokładna (dla symulacji, gdy znamy s)
        is_exact = np.array_equal(candidate_s, self.s)

        return is_correct or is_exact


def build_primal_lattice(A: np.ndarray, b: np.ndarray, q: int) -> IntegerMatrix:
    """
    Konstruuje macierz bazy dla ataku Primal (Kannan's Embedding).

    B = [ qI_m   0     0 ]
        [  A^T   I_n   0 ]
        [  b^T   0     1 ]
    """
    m, n = A.shape
    d = m + n + 1

    # Tworzymy macierz w fpylll (IntegerMatrix)
    B = IntegerMatrix(d, d)

    # Wypełnianie bloku q*I_m (lewy górny róg)
    for i in range(m):
        B[i, i] = q

    # Wypełnianie bloku A^T (lewy środek) i I_n (prawy środek)
    # A ma wymiary m x n, więc A^T ma n x m.
    # W fpylll indeksujemy B[wiersz, kolumna]
    for i in range(n):
        row_idx = m + i
        # Wstawianie wiersza z A^T (czyli kolumny z A)
        for j in range(m):
            B[row_idx, j] = int(A[j, i])  # A[j, i] to element A^T[i, j]

        # Wstawianie I_n (diagonalnie obok A^T)
        B[row_idx, m + i] = 1

    # Wypełnianie wektora b (ostatni wiersz)
    row_last = m + n
    for j in range(m):
        B[row_last, j] = int(b[j])

    # Wstawianie 1 na końcu (embedding factor)
    B[row_last, m + n] = 1

    return B


def primal_attack(
    A: np.ndarray, b: np.ndarray, q: int, block_size: int = 20, log: bool = True
) -> tuple[np.ndarray | None, float]:
    """
    Przeprowadza atak Primal używając redukcji BKZ.
    """
    m, n = A.shape
    print(f"[*] Rozpoczynanie ataku Primal.")
    print(f"[*] Wymiar kraty: {m + n + 1} x {m + n + 1}")

    # 1. Konstrukcja kraty
    start_time = time.time()
    B = build_primal_lattice(A, b, q)
    print(f"[+] Krata skonstruowana.")

    # 2. Redukcja LLL (wstępna)
    print(f"[*] Uruchamianie LLL...")
    LLL.reduction(B)

    # 3. Redukcja BKZ (silniejsza)
    print(f"[*] Uruchamianie BKZ-{block_size}...")
    BKZ.reduction(B, BKZ.Param(block_size))

    reduction_time = time.time() - start_time
    print(f"[+] Redukcja zakończona w {reduction_time:.2f}s.")

    # 4. Poszukiwanie sekretu w zredukowanej bazie
    # Spodziewamy się wektora v = (e, s, 1) lub v = (-e, -s, -1)
    # Sekret s znajduje się na indeksach [m : m+n]

    # Sprawdzamy kilka pierwszych najkrótszych wektorów
    for row_idx in range(min(10, B.nrows)):
        v = B[row_idx]

        # Pobierz potencjalny sekret (kandydat)
        # W fpylll wektory są indeksowane jak listy
        s_candidate = [v[j] for j in range(m, m + n)]
        s_candidate = np.array(s_candidate)

        # Sprawdzamy s
        # Musimy sprawdzić, czy działa dla oryginalnego równania modulo q
        # Z uwagi na modulo, ujemne wartości w Pythonie trzeba traktować ostrożnie,
        # ale tutaj s jest elementem Z_q, więc zrobimy modulo.

        s_candidate_mod = s_candidate % q
        if verify_candidate(A, b, s_candidate_mod, q):
            return s_candidate_mod, reduction_time

        # Sprawdzamy -s (ponieważ SVP jest z dokładnością do znaku)
        s_candidate_neg_mod = (-s_candidate) % q
        if verify_candidate(A, b, s_candidate_neg_mod, q):
            return s_candidate_neg_mod, reduction_time

    print("[-] Atak nie powiódł się. Spróbuj zwiększyć block_size lub liczbę próbek m.")
    return None, reduction_time


def verify_candidate(
    A: np.ndarray, b: np.ndarray, s_candidate: np.ndarray, q: int, alpha: float = 0.005
) -> bool:  # Dodaj alpha
    b_prime = A @ s_candidate % q
    diff = (b - b_prime + q // 2) % q - q // 2
    norm = np.linalg.norm(diff)

    # Obliczamy oczekiwaną normę dla PRAWDZIWEGO sekretu
    # m = len(b)
    expected_norm = alpha * q * np.sqrt(len(b))

    # Dajemy mały margines (np. 1.5x lub 2x), ale nie q/4!
    limit = expected_norm * 2.0

    return norm < limit


# --- GŁÓWNA FUNKCJA ---

if __name__ == "__main__":
    print("=========================================")
    print("   PROJEKT: ATAK PRIMAL NA LWE (DEMO)    ")
    print("=========================================")

    n = 10
    m = 60  # Zazwyczaj m = 2n dla wystarczającej nadmiarowości
    q = 101  # Liczba pierwsza
    alpha = 0.001  # Parametr szumu (sigma = alpha * q)

    # Parametr ataku
    bkz_block_size = 25  # Im większy, tym silniejszy atak, ale wolniejszy

    # 1. Generowanie Wyzwania
    lwe = LWEInstance(n, m, q, alpha)
    print(f"[*] Generowanie instancji LWE (n={n}, m={m}, q={q})...")
    A, b = lwe.generate()
    print("[+] Instancja wygenerowana pomyślnie.")

    # 2. Uruchomienie ataku
    recovered_s, time_taken = primal_attack(A, b, q, block_size=bkz_block_size)

    print(f"Atak zajął: {time_taken:.2f}s\n")

    # 3. Wyniki
    print("=========================================")
    print("                WYNIKI                   ")
    print("=========================================")

    if lwe.check_solution(recovered_s):
        print("[SUKCES] Znaleziono sekret!")
        print(f"Oryginalny s:   {lwe.s}")
        print(f"Odzyskany s:    {recovered_s}")

        # Ostateczna weryfikacja
        if np.array_equal(lwe.s, recovered_s):
            print("Weryfikacja: POPRAWNY (Idealna zgodność)")
        else:
            print("Weryfikacja: POPRAWNY (Matematycznie równoważny)")
    else:
        print("[PORAŻKA] Nie udało się odzyskać sekretu.")
        print("Sugestie:")
        print(" - Zwiększ parametr block_size w BKZ")
        print(" - Zwiększ liczbę próbek m")
        print(" - Zmniejsz szum (alpha)")

    print("=========================================")
