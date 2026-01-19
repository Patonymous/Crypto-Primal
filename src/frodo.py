"""
FrodoKEM - Implementacja uproszczona mechanizmu enkapsulacji klucza.

FrodoKEM to kryptosystem oparty na problemie LWE (Learning With Errors),
zaprojektowany do bezpieczeństwa postkwantowego. W przeciwieństwie do
Regev'a, FrodoKEM używa macierzy zamiast wektorów dla sekretów i błędów.

Implementacja oparta na:
- FrodoKEM specification (https://frodokem.org/)
- NIST PQC Round 3 submission
"""

import numpy as np
import json
import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional


def sample_error_matrix(rows: int, cols: int, sigma: float) -> np.ndarray:
    """
    Próbkuje macierz błędów z dyskretnego rozkładu Gaussa.
    W praktyce FrodoKEM używa tabelarycznego rozkładu, ale dla
    uproszczenia używamy zaokrąglonego rozkładu Gaussa.
    """
    return np.round(np.random.normal(0, sigma, size=(rows, cols))).astype(int)


def sample_uniform_matrix(rows: int, cols: int, q: int) -> np.ndarray:
    """Próbkuje macierz jednostajnie z Z_q."""
    return np.random.randint(0, q, size=(rows, cols))


def center_modulo(x: np.ndarray, q: int) -> np.ndarray:
    """Centruje wartości modulo q do zakresu [-q/2, q/2)."""
    return (x + q // 2) % q - q // 2


@dataclass
class FrodoParams:
    """Parametry FrodoKEM."""
    n: int          # Wymiar sekretu (liczba kolumn A)
    m: int          # Liczba próbek (liczba wierszy A) - dla ataku m > n
    n_bar: int      # Liczba kolumn macierzy S i E (typowo 8)
    m_bar: int      # Liczba wierszy macierzy S' i E' (typowo 8)
    q: int          # Moduł (potęga 2, np. 2^15 = 32768)
    sigma: float    # Odchylenie standardowe rozkładu błędów
    B: int          # Liczba bitów na element podczas enkodowania wiadomości
    
    @property
    def extracted_bits(self) -> int:
        """Liczba bitów w wyekstrahowanym kluczu współdzielonym."""
        return self.m_bar * self.n_bar * self.B


class FrodoKEM:
    """
    Uproszczona implementacja FrodoKEM dla celów demonstracyjnych.
    
    Schemat:
    - KeyGen: Generuje parę kluczy (pk, sk)
    - Encaps: Enkapsuluje losowy klucz współdzielony
    - Decaps: Dekapsuluje klucz współdzielony
    """
    
    def __init__(self, params: Optional[FrodoParams] = None):
        self.params = params
        
        # Klucz publiczny
        self.A: Optional[np.ndarray] = None      # Macierz publiczna (n x n)
        self.B_pub: Optional[np.ndarray] = None  # B = AS + E (n x n_bar)
        
        # Klucz prywatny
        self._S: Optional[np.ndarray] = None     # Sekret (n x n_bar)
        self._E: Optional[np.ndarray] = None     # Błąd (n x n_bar)
        
        # Seed dla generowania A (dla kompresji klucza)
        self._seed_A: Optional[bytes] = None
    
    def _generate_A_from_seed(self, seed: bytes) -> np.ndarray:
        """
        Generuje macierz A deterministycznie z seeda.
        W praktyce FrodoKEM używa SHAKE128, tutaj uproszczenie z hashlib.
        """
        m = self.params.m
        n = self.params.n
        q = self.params.q
        
        # Użyj seeda do inicjalizacji generatora
        np.random.seed(int.from_bytes(hashlib.sha256(seed).digest()[:4], 'big'))
        A = np.random.randint(0, q, size=(m, n))
        
        return A
    
    def keygen(self) -> None:
        """
        Generuje parę kluczy FrodoKEM.
        
        Klucz publiczny: (seed_A, B) gdzie B = AS + E mod q
        Klucz prywatny: S
        """
        if self.params is None:
            raise ValueError("Parametry nie zostały ustawione.")
        
        n = self.params.n
        m = self.params.m
        n_bar = self.params.n_bar
        q = self.params.q
        sigma = self.params.sigma
        
        # Generuj seed dla A
        self._seed_A = secrets.token_bytes(16)
        
        # Generuj macierz A z seeda (m x n - więcej wierszy niż kolumn dla ataku)
        self.A = self._generate_A_from_seed(self._seed_A)
        
        # Próbkuj sekret S i błąd E
        self._S = sample_error_matrix(n, n_bar, sigma)
        self._E = sample_error_matrix(m, n_bar, sigma)
        
        # Oblicz B = AS + E mod q (B ma wymiary m x n_bar)
        self.B_pub = (self.A @ self._S + self._E) % q
    
    def encaps(self, mu: Optional[bytes] = None) -> tuple[tuple[np.ndarray, np.ndarray], bytes]:
        """
        Enkapsuluje klucz współdzielony.
        
        Zwraca:
            ciphertext: (C1, C2) gdzie:
                - C1 = S'A + E' (m_bar x n)
                - C2 = S'B + E'' + encode(mu) (m_bar x n_bar)
            shared_secret: Klucz współdzielony (bajty)
        """
        if self.A is None or self.B_pub is None:
            raise ValueError("Klucz publiczny nie został wygenerowany.")
        
        n = self.params.n
        m = self.params.m
        n_bar = self.params.n_bar
        m_bar = self.params.m_bar
        q = self.params.q
        sigma = self.params.sigma
        B_bits = self.params.B
        
        # Oblicz liczbę bitów możliwych do zakodowania
        total_bits = m_bar * n_bar * B_bits
        mu_bytes = (total_bits + 7) // 8
        
        # Generuj losową wiadomość mu (jeśli nie podano)
        if mu is None:
            mu = secrets.token_bytes(mu_bytes)
        
        # Obetnij mu do dokładnie mu_bytes (jeśli dłuższe)
        mu = mu[:mu_bytes]
        
        # Próbkuj S', E', E''
        S_prime = sample_error_matrix(m_bar, m, sigma)  # m_bar x m (bo A jest m x n)
        E_prime = sample_error_matrix(m_bar, n, sigma)
        E_double_prime = sample_error_matrix(m_bar, n_bar, sigma)
        
        # C1 = S'A + E' mod q (m_bar x n)
        C1 = (S_prime @ self.A + E_prime) % q
        
        # Enkoduj mu do macierzy
        mu_encoded = self._encode_message(mu, m_bar, n_bar, q, B_bits)
        
        # C2 = S'B + E'' + encode(mu) mod q (m_bar x n_bar)
        C2 = (S_prime @ self.B_pub + E_double_prime + mu_encoded) % q
        
        # Klucz współdzielony to hash mu
        shared_secret = hashlib.sha256(mu).digest()
        
        return (C1, C2), shared_secret
    
    def decaps(self, ciphertext: tuple[np.ndarray, np.ndarray]) -> bytes:
        """
        Dekapsuluje klucz współdzielony.
        
        Args:
            ciphertext: (C1, C2)
            
        Zwraca:
            shared_secret: Klucz współdzielony (bajty)
        """
        if self._S is None:
            raise ValueError("Klucz prywatny nie jest dostępny.")
        
        C1, C2 = ciphertext
        q = self.params.q
        m_bar = self.params.m_bar
        n_bar = self.params.n_bar
        B_bits = self.params.B
        
        # M = C2 - C1 * S mod q
        # M ≈ S'B + E'' + encode(mu) - (S'A + E')S
        #   = S'(AS + E) + E'' + encode(mu) - S'AS - E'S
        #   = S'E + E'' - E'S + encode(mu)
        #   ≈ encode(mu) (jeśli błędy są małe)
        M = (C2 - C1 @ self._S) % q
        
        # Dekoduj wiadomość
        mu = self._decode_message(M, m_bar, n_bar, q, B_bits)
        
        # Klucz współdzielony to hash mu
        shared_secret = hashlib.sha256(mu).digest()
        
        return shared_secret
    
    def _encode_message(self, mu: bytes, m_bar: int, n_bar: int, q: int, B: int) -> np.ndarray:
        """
        Enkoduje wiadomość mu do macierzy m_bar x n_bar.
        Każdy element przechowuje B bitów.
        Używamy tylko pierwszych (m_bar * n_bar * B) bitów z mu.
        """
        encoded = np.zeros((m_bar, n_bar), dtype=int)
        total_bits = m_bar * n_bar * B
        
        # Konwertuj bajty na bity (MSB first)
        bits = []
        for byte in mu:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
                if len(bits) >= total_bits:
                    break
            if len(bits) >= total_bits:
                break
        
        # Dopełnij zerami jeśli za mało bitów
        while len(bits) < total_bits:
            bits.append(0)
        
        # Wypełnij macierz - każdy element zawiera B bitów
        bit_idx = 0
        scale = q // (1 << B)  # Skalowanie dla B bitów
        
        for i in range(m_bar):
            for j in range(n_bar):
                value = 0
                for b in range(B):
                    value = (value << 1) | bits[bit_idx]
                    bit_idx += 1
                encoded[i, j] = value * scale
        
        return encoded
    
    def _decode_message(self, M: np.ndarray, m_bar: int, n_bar: int, q: int, B: int) -> bytes:
        """
        Dekoduje macierz M do wiadomości mu.
        Wartości w M mogą być zaszumione, więc używamy zaokrąglenia.
        """
        total_bits = m_bar * n_bar * B
        bits = []
        scale = q // (1 << B)
        
        for i in range(m_bar):
            for j in range(n_bar):
                # Wartość może być w zakresie [0, q) 
                value = int(M[i, j])
                
                # Dekoduj B bitów przez zaokrąglenie
                # Dodaj połowę scale dla lepszego zaokrąglenia
                decoded_value = ((value + scale // 2) // scale) % (1 << B)
                
                for b in range(B - 1, -1, -1):
                    bits.append((decoded_value >> b) & 1)
        
        # Konwertuj bity na bajty (MSB first, z paddingiem do pełnych bajtów)
        num_bytes = (total_bits + 7) // 8
        result = bytearray(num_bytes)
        
        for byte_idx in range(num_bytes):
            byte = 0
            for bit_pos in range(8):
                bit_idx = byte_idx * 8 + bit_pos
                if bit_idx < len(bits):
                    byte = (byte << 1) | bits[bit_idx]
                else:
                    byte = byte << 1  # Pad with zeros
            result[byte_idx] = byte
        
        return bytes(result)
    
    def save(self, filename: str, private: bool) -> bool:
        """Zapisuje klucze do pliku."""
        try:
            with open(filename, "w") as writer:
                data = {
                    "params": {
                        "n": self.params.n,
                        "m": self.params.m,
                        "n_bar": self.params.n_bar,
                        "m_bar": self.params.m_bar,
                        "q": self.params.q,
                        "sigma": self.params.sigma,
                        "B": self.params.B
                    },
                    "seed_A": self._seed_A.hex() if self._seed_A else None,
                    "A": self.A.tolist() if self.A is not None else None,
                    "B_pub": self.B_pub.tolist() if self.B_pub is not None else None,
                }
                if private:
                    data["S"] = self._S.tolist() if self._S is not None else None
                    data["E"] = self._E.tolist() if self._E is not None else None
                writer.write(json.dumps(data, indent=2))
            return True
        except Exception:
            return False
    
    def load(self, filename: str, private: bool) -> bool:
        """Wczytuje klucze z pliku."""
        try:
            with open(filename, "r") as reader:
                data = json.load(reader)
                
                params_data = data["params"]
                self.params = FrodoParams(
                    n=params_data["n"],
                    m=params_data["m"],
                    n_bar=params_data["n_bar"],
                    m_bar=params_data["m_bar"],
                    q=params_data["q"],
                    sigma=params_data["sigma"],
                    B=params_data["B"]
                )
                
                self._seed_A = bytes.fromhex(data["seed_A"]) if data.get("seed_A") else None
                self.A = np.array(data["A"]) if data.get("A") else None
                self.B_pub = np.array(data["B_pub"]) if data.get("B_pub") else None
                
                if private:
                    self._S = np.array(data["S"]) if data.get("S") else None
                    self._E = np.array(data["E"]) if data.get("E") else None
            return True
        except Exception:
            return False
    
    def get_lwe_instance_for_attack(self) -> tuple[np.ndarray, np.ndarray, int, float]:
        """
        Zwraca instancję LWE odpowiadającą kluczowi publicznemu FrodoKEM.
        
        FrodoKEM można atakować jako problem LWE:
        B = AS + E mod q
        
        Gdzie:
        - A jest macierzą m x n (więcej wierszy niż kolumn)
        - S jest macierzą n x n_bar (sekret)
        - E jest macierzą m x n_bar (błąd)
        - B jest macierzą m x n_bar (publiczna)
        
        Dla ataku primal, atakujemy kolumna po kolumnie:
        b_i = A s_i + e_i mod q dla i = 1, ..., n_bar
        
        Zwracamy pierwszą kolumnę jako reprezentatywną instancję.
        """
        if self.A is None or self.B_pub is None:
            raise ValueError("Klucz publiczny nie został wygenerowany.")
        
        # Zwróć pierwszą kolumnę jako instancję LWE
        # b = A @ s + e gdzie s = S[:, 0], e = E[:, 0]
        A = self.A
        b = self.B_pub[:, 0]
        q = self.params.q
        sigma = self.params.sigma
        
        return A, b, q, sigma
    
    def check_secret_recovery(self, recovered_S: np.ndarray) -> bool:
        """Sprawdza, czy odzyskany sekret jest poprawny."""
        if self._S is None:
            raise ValueError("Klucz prywatny nie jest dostępny do weryfikacji.")
        
        return np.array_equal(recovered_S, self._S)
    
    def check_column_recovery(self, col_idx: int, recovered_s: np.ndarray) -> bool:
        """Sprawdza, czy odzyskana kolumna sekretu jest poprawna."""
        if self._S is None:
            raise ValueError("Klucz prywatny nie jest dostępny do weryfikacji.")
        
        return np.array_equal(recovered_s, self._S[:, col_idx])


# Predefiniowane zestawy parametrów (uproszczone dla demonstracji)

def get_demo_params() -> FrodoParams:
    """
    Parametry demonstracyjne - bardzo słabe, ale szybkie do ataku.
    NIE UŻYWAĆ W PRODUKCJI!
    
    Aby atak primal działał efektywnie, potrzebujemy:
    - Małe n (wymiar sekretu)
    - m > n (więcej próbek niż wymiar sekretu)
    - Bardzo małe sigma (poziom szumu)
    - Mały moduł q
    """
    return FrodoParams(
        n=10,        # Wymiar sekretu (małe dla demonstracji)
        m=60,        # Liczba próbek (m >> n dla skutecznego ataku)
        n_bar=2,     # Mała liczba kolumn
        m_bar=2,     # Mała liczba wierszy
        q=251,       # Większy moduł dla lepszej dekapsulacji
        sigma=0.5,   # Bardzo małe odchylenie 
        B=2          # 2 bity na element (scale = q/4 = 62)
    )


def get_toy_params() -> FrodoParams:
    """
    Parametry 'zabawkowe' - słabe, ale nieco trudniejsze niż demo.
    NIE UŻYWAĆ W PRODUKCJI!
    """
    return FrodoParams(
        n=15,
        m=80,        # m > n
        n_bar=4,
        m_bar=4,
        q=251,       # Większy moduł
        sigma=2.0,
        B=2
    )


def get_frodo640_like_params() -> FrodoParams:
    """
    Parametry inspirowane FrodoKEM-640 - ZREDUKOWANE dla demonstracji.
    Prawdziwy FrodoKEM-640 ma n=640, q=2^15.
    
    Te parametry są znacznie słabsze niż prawdziwy FrodoKEM-640!
    """
    return FrodoParams(
        n=32,
        m=128,       # m > n
        n_bar=8,
        m_bar=8,
        q=32749,     # Duży moduł (pierwsza liczba < 2^15)
        sigma=2.8,
        B=2
    )
