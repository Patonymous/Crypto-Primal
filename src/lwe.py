import numpy as np


def center_modulo(x: np.ndarray, q: int) -> np.ndarray:
    return (x + q // 2) % q - q // 2


class LWE:
    """
    Klasa pomocnicza przechowująca instancję problemu LWE.
    """

    def __init__(self, n: int, m: int, q: int, alpha: float):
        self.n = n  # Wymiar sekretu
        self.m = m  # Liczba próbek
        self.q = q  # Moduł
        self.sigma = alpha * q  # Odchylenie standardowe szumu

        # Publicznie znane
        self.A = None
        self.b = None

        # Tajne
        self._s = None
        self._e = None

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Generuje losową instancję LWE: b = As + e mod q."""

        # 1. Macierz publiczna A (m x n)
        self.A = np.random.randint(0, self.q, size=(self.m, self.n))

        # 2. Sekret s (n) - losowy jednostajny
        self._s = np.random.randint(0, self.q, size=self.n)

        # 3. Błąd e (m) - rozkład Gaussa, zaokrąglony do liczb całkowitych
        # W praktyce LWE często używa dyskretnego rozkładu Gaussa,
        # to jest przybliżenie, ale dostateczne na potrzeby projektu.
        self._e = np.round(np.random.normal(0, self.sigma, size=self.m)).astype(int)

        self.b = (self.A @ self._s + self._e) % self.q

        return self.A, self.b

    def check_solution_correct(
        self, candidate_e: np.ndarray, candidate_s: np.ndarray, margin: float = 2.0
    ) -> bool:
        """Weryfikuje, czy podany kandydat na sekret jest poprawny."""

        if self.A is None or self.b is None:
            raise ValueError("Instancja LWE nie została wygenerowana.")

        # Sprawdź czy As jest blisko b (różnica powinna być małym błędem)
        b_calc = self.A @ candidate_s % self.q
        diff = center_modulo(self.b - b_calc, self.q)

        # Jeśli norma różnicy jest mała, rozwiązanie jest poprawne
        norm_diff = np.linalg.norm(diff)
        expected_norm = np.sqrt(self.m) * self.sigma

        # Margines błędu dla weryfikacji - domyślnie 2x oczekiwanej normy
        return norm_diff < (expected_norm * margin)

    def check_solution_exact(self, candidate_s: np.ndarray | None) -> bool:
        """Weryfikuje, czy podany kandydat na sekret jest dokładnie taki sam jak sekret."""

        if candidate_s is None:
            return False

        return np.array_equal(candidate_s, self._s)
