import numpy as np
import json


def center_modulo(x: np.ndarray, q: int) -> np.ndarray:
    return (x + q // 2) % q - q // 2


class LWE:
    """
    Klasa pomocnicza przechowująca instancję problemu LWE.
    """

    def __init__(self):
        self.n: int  # Wymiar sekretu
        self.m: int  # Liczba próbek
        self.q: int  # Moduł
        self.sigma: float  # Odchylenie standardowe szumu

        # Publicznie znane
        self.A: np.ndarray = None
        self.b: np.ndarray = None

        # Tajne
        self._s: np.ndarray | None = None
        self._e: np.ndarray | None = None

    def generate(self, n: int, m: int, q: int, alpha: float) -> None:
        """Generuje losową instancję LWE: b = As + e mod q."""

        self.n = n
        self.m = m
        self.q = q
        self.sigma = alpha * q

        # Macierz publiczna A (m x n)
        self.A = np.random.randint(0, q, size=(m, n))

        # Sekret s (n) - losowy jednostajny
        self._s = np.random.randint(0, q, size=n)

        # Błąd e (m) - rozkład Gaussa, zaokrąglony do liczb całkowitych
        # W praktyce LWE często używa dyskretnego rozkładu Gaussa,
        # to jest przybliżenie, ale dostateczne na potrzeby projektu.
        self._e = np.round(np.random.normal(0, self.sigma, size=m)).astype(int)

        self.b = (self.A @ self._s + self._e) % q

    def save(self, filename: str, private: bool) -> bool:
        try:
            with open(filename, "w") as writer:
                data = {
                    "n": self.n,
                    "m": self.m,
                    "q": self.q,
                    "sigma": self.sigma,
                    "A": self.A.tolist(),
                    "b": self.b.tolist(),
                }
                if private:
                    data["s"] = self._s.tolist()
                    data["e"] = self._e.tolist()
                writer.write(json.dumps(data, indent=2))
        except Exception:
            return False
        return True

    def load(self, filename: str, private: bool) -> bool:
        try:
            with open(filename, "r") as reader:
                data = json.load(reader)
                self.n = int(data["n"])
                self.m = int(data["m"])
                self.q = int(data["q"])
                self.sigma = float(data["sigma"])
                self.A = np.array(data["A"])
                self.b = np.array(data["b"])
                if private:
                    self._s = np.array(data["s"])
                    self._e = np.array(data["e"])
        except Exception:
            return False
        return True

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
