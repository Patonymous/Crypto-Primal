#!/usr/bin/env python3

import sys
import typing
import numpy as np
from src.printer import Printer, Style
from src.lwe import LWE
from src.attack import primal_attack
from src.regev import encrypt, decrypt
from src.interactive import interactive_attack
from src.benchmark import compare_block_sizes, print_comparison_table, compare_parameters, print_params_comparison_table
from src.visualization import visualize_reduction, visualize_block_size_impact, plot_success_heatmap


def save_lwe(lwe: LWE, private: bool) -> None:
    filename = "lwe.key" if private else "lwe.pub"
    success = lwe.save(filename, private)
    if not success:
        with Printer(Style.ERROR) as printer:
            printer("")
            printer(f"Nie udało się zapisać instancji LWE do pliku {filename}.")
        sys.exit(1)


def load_lwe(private: bool) -> LWE:
    filename = "lwe.key" if private else "lwe.pub"
    lwe = LWE()
    success = lwe.load(filename, private)
    if not success:
        with Printer(Style.ERROR) as printer:
            printer("")
            printer(f"Nie udało się wczytać instancji LWE z pliku {filename}.")
            printer("Czy uruchomiono wcześniej skrypt z opcją 'generuj'?.")
        sys.exit(1)
    return lwe


def generate(n: int, m: int, q: int, alpha: float) -> None:
    with Printer(Style.INFO) as printer:
        printer(f"Wybrane parametry: n={n}, m={m}, q={q}, alpha={alpha}")

    with Printer(Style.LOG) as printer:
        printer("Generowanie instancji LWE...", end="")
    lwe = LWE()
    lwe.generate(n, m, q, alpha)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.LOG) as printer:
        printer("Zapisywanie instancji LWE...", end="")
    save_lwe(lwe, False)
    save_lwe(lwe, True)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.INFO) as printer:
        printer(
            "Instancja LWE została wygenerowana i zapisana do plików 'lwe.pub' i 'lwe.key'."
        )
    return lwe


def save_ciphertext(ciphertexts: list[tuple[np.ndarray, int]]) -> None:
    try:
        with open("ciphertext.bin", "wb") as writer:
            Ss = [ciphertext[0] for ciphertext in ciphertexts]
            cs = [ciphertext[1] for ciphertext in ciphertexts]
            np.savez_compressed(writer, Ss=Ss, cs=cs)
    except Exception:
        with Printer(Style.ERROR) as printer:
            printer("")
            printer("Nie udało się zapisać szyfrowanej wiadomości.")
        sys.exit(1)


def load_ciphertext() -> list[tuple[np.ndarray, int]]:
    try:
        with open("ciphertext.bin", "rb") as reader:
            data = np.load(reader)
            Ss: list[np.ndarray] = data["Ss"]
            cs: list[int] = data["cs"]
            return [(S, c) for S, c in zip(Ss, cs)]
    except Exception:
        with Printer(Style.ERROR) as printer:
            printer("")
            printer(
                "Nie udało się wczytać szyfrowanej wiadomości z pliku 'ciphertext.bin'."
            )
            printer("Czy uruchomiono wcześniej skrypt z opcją 'szyfruj'?.")
        sys.exit(1)


def encrypt_message(message: str) -> None:
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie instancji LWE...", end="")
    lwe = load_lwe(False)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.LOG) as printer:
        printer("Szyfrowanie wiadomości...", end="")
    ciphertexts = encrypt(lwe, message)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.LOG) as printer:
        printer("Zapisywanie szyfrowanej wiadomości...", end="")
    save_ciphertext(ciphertexts)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.INFO) as printer:
        printer("Zaszyfrowana wiadomość została zapisana do pliku 'ciphertext.bin'.")


def decrypt_message() -> None:
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie instancji LWE...", end="")
    lwe = load_lwe(True)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.LOG) as printer:
        printer("Wczytywanie szyfrowanej wiadomości...", end="")
    ciphertexts = load_ciphertext()
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.LOG) as printer:
        printer("Odszyfrowywanie wiadomości...", end="")
    decrypted_message = decrypt(lwe, ciphertexts)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.INFO) as printer:
        printer(f"Odszyfrowana wiadomość: {decrypted_message}")


def interactive(bkz_block_size: int) -> None:
    """Tryb interaktywny ataku."""
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie instancji LWE...", end="")
    lwe = load_lwe(True)  # Potrzebujemy sekretu do weryfikacji
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    interactive_attack(lwe, bkz_block_size, auto=False)


def benchmark(trials: int) -> None:
    """Benchmark porównujący rozmiary bloków BKZ i parametry LWE."""
    block_sizes = [10, 15, 20, 25, 30]
    
    # Część 1: Porównanie rozmiarów bloków dla aktualnej instancji LWE
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie instancji LWE...", end="")
    lwe = load_lwe(False)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    alpha = lwe.sigma / lwe.q
    
    with Printer(Style.INFO) as printer:
        printer("")
        printer("=" * 60)
        printer("CZĘŚĆ 1: PORÓWNANIE ROZMIARÓW BLOKÓW BKZ")
        printer("=" * 60)
    
    results_blocks = compare_block_sizes(
        n=lwe.n, m=lwe.m, q=lwe.q, alpha=alpha,
        block_sizes=block_sizes, trials=trials
    )
    print_comparison_table(results_blocks)
    
    # Część 2: Porównanie różnych parametrów LWE
    with Printer(Style.INFO) as printer:
        printer("")
        printer("=" * 60)
        printer("CZĘŚĆ 2: PORÓWNANIE PARAMETRÓW LWE")
        printer("=" * 60)
    
    param_sets = [
        {'n': 8, 'm': 50, 'q': 97, 'alpha': 0.01},
        {'n': 10, 'm': 60, 'q': 101, 'alpha': 0.01},
        {'n': 12, 'm': 70, 'q': 127, 'alpha': 0.01},
        {'n': 10, 'm': 60, 'q': 101, 'alpha': 0.005},
        {'n': 10, 'm': 60, 'q': 101, 'alpha': 0.02},
        {'n': 10, 'm': 80, 'q': 101, 'alpha': 0.01},
        {'n': 10, 'm': 60, 'q': 199, 'alpha': 0.01},
    ]
    
    results_params = compare_parameters(param_sets, block_size=25, trials=trials)
    print_params_comparison_table(results_params)


def visualize() -> None:
    """Wizualizacja redukcji kraty."""
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie instancji LWE...", end="")
    lwe = load_lwe(False)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    with Printer(Style.INFO) as printer:
        printer("Generowanie wykresów...")
    
    visualize_reduction(lwe, block_sizes=[10, 20, 30], save_path="reduction_quality.png")
    visualize_block_size_impact(lwe, max_block=40, save_path="block_size_impact.png")
    
    with Printer(Style.SUCCESS) as printer:
        printer("Wykresy zapisane do plików:")
        printer("  - reduction_quality.png")
        printer("  - block_size_impact.png")


def attack_message(bkz_block_size: int) -> None:
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie instancji LWE...", end="")
    lwe = load_lwe(False)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.INFO) as printer:
        printer(f"Atak rozpoczęty, rozmiar bloku BKZ: {bkz_block_size}")

    recovered_s, time_taken = primal_attack(lwe, bkz_block_size)

    with Printer(Style.INFO) as printer:
        printer(f"Atak zakończony w {time_taken:.2f}s")

    if recovered_s is None:
        with Printer(Style.FAIL) as printer:
            printer("Nie udało się odgadnąć sekretu.")
        return

    with Printer(Style.SUCCESS) as printer:
        printer(f"Odgadnięto sekret: {recovered_s}")

    lwe._s = recovered_s

    with Printer(Style.LOG) as printer:
        printer("Wczytywanie szyfrowanej wiadomości...", end="")
    ciphertexts = load_ciphertext()
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.LOG) as printer:
        printer("Odszyfrowywanie wiadomości...", end="")
    decrypted_message = decrypt(lwe, ciphertexts)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    with Printer(Style.INFO) as printer:
        printer(f"Odszyfrowana wiadomość: {decrypted_message}")


def get_param(
    index: int,
    type: typing.Type[int | float | str],
    name: str,
    default: int | float | str | None,
) -> int | float | str | None:
    if len(sys.argv) > index:
        try:
            if type == str:
                return sys.argv[index]
            elif type == int:
                return int(sys.argv[index])
            elif type == float:
                return float(sys.argv[index])
            else:
                raise ValueError(f"Niepoprawny typ: {type}")
        except Exception as e:
            with Printer(Style.ERROR) as printer:
                printer(f"Niepoprawna wartość parametru '{name}': {e}")
            sys.exit(1)
    if default is not None:
        with Printer(Style.LOG) as printer:
            printer(f"Przyjęto domyślną wartość parametru '{name}': {default}")
    return default


def is_prime(num: int) -> bool:
    if num < 2:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(3, int(num**0.5) + 1, 2):
        if num % i == 0:
            return False
    return True


def main():
    with Printer(Style.BANNER) as printer:
        printer("=======================================")
        printer("          ATAK PRIMAL NA LWE           ")
        printer("     NA PRZYKŁADZIE KRYPTOSYSTEMU      ")
        printer("    KLUCZA PUBLICZNEGO REGEV (2005)    ")
        printer("=======================================")

    action = get_param(1, str, "akcja", default=None)
    if action is None:
        with Printer(Style.ERROR) as printer:
            printer("Akcja jest wymagana")
            printer(
                # cspell: disable-next-line
                "Dostępne akcje: p[omoc], g[eneruj], s[zyfruj], o[dszyfruj] lub a[takuj]"
            )
        sys.exit(1)

    if "pomoc".startswith(action) or action in ["-h", "--help"]:
        with Printer(Style.INFO) as printer:
            # cspell: disable
            printer("Dostępne akcje:")
            printer("  p[omoc] - wyświetla tę pomoc")
            printer("  g[eneruj] - generuje nową instancję LWE")
            printer("    Powinna być wywołana jako pierwsza akcja.")
            printer("    Przyjmuje opcjonalne parametry pozycyjne: n, m, q, alpha")
            printer("    Domyślne wartości: n=10, m=60, q=101, alpha=0.01")
            printer("  s[zyfruj] - szyfruje wiadomość")
            printer("    Powinna być wywołana po akcji 'generuj'.")
            printer("    Przyjmuje wymagany parametr: wiadomość")
            printer("  o[dszyfruj] - odszyfrowuje wiadomość")
            printer("    Powinna być wywołana po akcji 'szyfruj'.")
            printer("  a[takuj] - atakuje wiadomość")
            printer("    Powinna być wywołana po akcji 'szyfruj'.")
            printer("    Przyjmuje opcjonalny parametr: rozmiar bloku BKZ")
            printer("    Domyślna wartość: 25")
            printer("  i[nteraktywny] - atak krok po kroku z wyjaśnieniami")
            printer("    Idealny do nauki i prezentacji.")
            printer("    Przyjmuje opcjonalny parametr: rozmiar bloku BKZ")
            printer("    Domyślna wartość: 25")
            printer("  b[enchmark] - porównanie skuteczności ataku")
            printer("    Testuje różne rozmiary bloków BKZ oraz parametry LWE.")
            printer("    Przyjmuje opcjonalny parametr: liczba prób (domyślnie 3)")
            printer("  w[izualizacja] - generuje wykresy jakości redukcji kraty")
            printer("    Wymaga matplotlib. Zapisuje wykresy do plików PNG.")
            printer("")
            printer("Opis parametrów:")
            printer(
                "  n - rozmiar sekretu, zwiększenie zmniejsza szansę powodzenia ataku"
            )
            printer(
                "  m - liczba próbek, zwiększenie zwiększa szansę powodzenia ataku, ale też czas trwania"
            )
            printer(
                "  q - moduł, powinien być liczbą pierwszą, zwiększenie zmniejsza szansę powodzenia ataku"
            )
            printer(
                "  alpha - poziom szumu, zwiększenie zmniejsza szansę powodzenia ataku"
            )
            printer(
                "  rozmiar bloku - siła redukcji kraty, zwiększenie zwiększa szansę powodzenia ataku, ale też czas trwania"
            )
            # cspell: enable

    elif "generuj".startswith(action):
        n = get_param(2, int, "n", default=10)
        m = get_param(3, int, "m", default=60)
        q = get_param(4, int, "q", default=101)

        if not is_prime(q):
            with Printer(Style.ERROR) as printer:
                printer(f"Parametr q musi być liczbą pierwszą, otrzymano: {q}")
            sys.exit(1)

        alpha = get_param(5, float, "alpha", default=0.01)
        generate(n, m, q, alpha)

    elif "szyfruj".startswith(action):
        message = get_param(2, str, "wiadomość", default=None)
        if message is None:
            with Printer(Style.ERROR) as printer:
                printer("Wiadomość jest wymagana")
            sys.exit(1)
        encrypt_message(message)

    elif "odszyfruj".startswith(action):
        decrypt_message()

    elif "atakuj".startswith(action):
        bkz_block_size = get_param(2, int, "rozmiar bloku BKZ", default=25)
        attack_message(bkz_block_size)

    elif "interaktywny".startswith(action):
        bkz_block_size = get_param(2, int, "rozmiar bloku BKZ", default=25)
        interactive(bkz_block_size)

    elif "benchmark".startswith(action):
        trials = get_param(2, int, "liczba prób", default=3)
        benchmark(trials)

    elif "wizualizacja".startswith(action):
        visualize()

    else:
        with Printer(Style.ERROR) as printer:
            printer(
                # cspell: disable-next-line
                f"Nieznana akcja: {action}. Użyj 'pomoc' aby zobaczyć dostępne opcje."
            )
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        with Printer(Style.WARNING) as printer:
            printer("")
            printer("Przerwano działanie programu.")
        sys.exit(0)
    except Exception as e:
        with Printer(Style.ERROR) as printer:
            printer("")
            printer(f"Wystąpił błąd: {e}")
        sys.exit(1)
