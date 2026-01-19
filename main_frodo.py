#!/usr/bin/env python3
"""
Główny skrypt demonstracyjny dla ataku Primal na FrodoKEM.

FrodoKEM to mechanizm enkapsulacji klucza (KEM) oparty na problemie LWE,
zaprojektowany z myślą o bezpieczeństwie postkwantowym.

Ten skrypt demonstruje:
1. Generowanie kluczy FrodoKEM
2. Enkapsulację i dekapsulację klucza współdzielonego
3. Atak Primal na FrodoKEM (odzyskanie sekretu)

UWAGA: Implementacja używa zredukowanych parametrów dla celów demonstracyjnych.
       Prawdziwy FrodoKEM z pełnymi parametrami jest odporny na te ataki.
"""

import sys
import typing
import numpy as np
from src.printer import Printer, Style
from src.frodo import (
    FrodoKEM, FrodoParams,
    get_demo_params, get_toy_params, get_frodo640_like_params
)
from src.attack_frodo import primal_attack_frodo, demo_key_recovery_attack


def save_frodo(frodo: FrodoKEM, private: bool) -> None:
    """Zapisuje klucze FrodoKEM do pliku."""
    filename = "frodo.key" if private else "frodo.pub"
    success = frodo.save(filename, private)
    if not success:
        with Printer(Style.ERROR) as printer:
            printer("")
            printer(f"Nie udało się zapisać kluczy FrodoKEM do pliku {filename}.")
        sys.exit(1)


def load_frodo(private: bool) -> FrodoKEM:
    """Wczytuje klucze FrodoKEM z pliku."""
    filename = "frodo.key" if private else "frodo.pub"
    frodo = FrodoKEM()
    success = frodo.load(filename, private)
    if not success:
        with Printer(Style.ERROR) as printer:
            printer("")
            printer(f"Nie udało się wczytać kluczy FrodoKEM z pliku {filename}.")
            printer("Czy uruchomiono wcześniej skrypt z opcją 'generuj'?")
        sys.exit(1)
    return frodo


def save_ciphertext(ciphertext: tuple[np.ndarray, np.ndarray], shared_secret: bytes) -> None:
    """Zapisuje szyfrogram do pliku."""
    try:
        C1, C2 = ciphertext
        with open("frodo_ciphertext.bin", "wb") as writer:
            np.savez_compressed(writer, C1=C1, C2=C2, shared_secret=shared_secret)
    except Exception:
        with Printer(Style.ERROR) as printer:
            printer("")
            printer("Nie udało się zapisać szyfrogramu.")
        sys.exit(1)


def load_ciphertext() -> tuple[tuple[np.ndarray, np.ndarray], bytes]:
    """Wczytuje szyfrogram z pliku."""
    try:
        with open("frodo_ciphertext.bin", "rb") as reader:
            data = np.load(reader)
            C1 = data["C1"]
            C2 = data["C2"]
            shared_secret = bytes(data["shared_secret"])
            return (C1, C2), shared_secret
    except Exception:
        with Printer(Style.ERROR) as printer:
            printer("")
            printer("Nie udało się wczytać szyfrogramu z pliku 'frodo_ciphertext.bin'.")
            printer("Czy uruchomiono wcześniej skrypt z opcją 'enkapsuluj'?")
        sys.exit(1)


def get_params_by_name(name: str) -> FrodoParams:
    """Zwraca parametry FrodoKEM na podstawie nazwy."""
    params_map = {
        "demo": get_demo_params,
        "toy": get_toy_params,
        "frodo640": get_frodo640_like_params,
    }
    
    if name not in params_map:
        with Printer(Style.ERROR) as printer:
            printer(f"Nieznany zestaw parametrów: {name}")
            printer(f"Dostępne: {', '.join(params_map.keys())}")
        sys.exit(1)
    
    return params_map[name]()


def generate(params_name: str) -> FrodoKEM:
    """Generuje nowe klucze FrodoKEM."""
    params = get_params_by_name(params_name)
    
    with Printer(Style.INFO) as printer:
        printer(f"Zestaw parametrów: {params_name}")
        printer(f"  n={params.n}, m={params.m}, n_bar={params.n_bar}, m_bar={params.m_bar}")
        printer(f"  q={params.q}, sigma={params.sigma}, B={params.B}")
    
    with Printer(Style.LOG) as printer:
        printer("Generowanie kluczy FrodoKEM...", end="")
    
    frodo = FrodoKEM(params)
    frodo.keygen()
    
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    with Printer(Style.LOG) as printer:
        printer("Zapisywanie kluczy...", end="")
    save_frodo(frodo, False)
    save_frodo(frodo, True)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    with Printer(Style.INFO) as printer:
        printer("Klucze FrodoKEM zostały wygenerowane i zapisane do plików:")
        printer("  'frodo.pub' (klucz publiczny)")
        printer("  'frodo.key' (klucz prywatny)")
    
    return frodo


def encapsulate() -> None:
    """Wykonuje enkapsulację klucza."""
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie klucza publicznego...", end="")
    frodo = load_frodo(False)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    with Printer(Style.LOG) as printer:
        printer("Enkapsulacja klucza współdzielonego...", end="")
    ciphertext, shared_secret = frodo.encaps()
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    with Printer(Style.LOG) as printer:
        printer("Zapisywanie szyfrogramu...", end="")
    save_ciphertext(ciphertext, shared_secret)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    with Printer(Style.INFO) as printer:
        printer(f"Klucz współdzielony (hex): {shared_secret.hex()}")
        printer("Szyfrogram zapisano do pliku 'frodo_ciphertext.bin'")


def decapsulate() -> None:
    """Wykonuje dekapsulację klucza."""
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie klucza prywatnego...", end="")
    frodo = load_frodo(True)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie szyfrogramu...", end="")
    ciphertext, original_shared_secret = load_ciphertext()
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    with Printer(Style.LOG) as printer:
        printer("Dekapsulacja klucza współdzielonego...", end="")
    decapsulated_secret = frodo.decaps(ciphertext)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    with Printer(Style.INFO) as printer:
        printer(f"Zdekapsulowany klucz (hex): {decapsulated_secret.hex()}")
    
    if decapsulated_secret == original_shared_secret:
        with Printer(Style.SUCCESS) as printer:
            printer("Dekapsulacja POPRAWNA - klucze są zgodne!")
    else:
        with Printer(Style.FAIL) as printer:
            printer("Dekapsulacja NIEPOPRAWNA - klucze różnią się!")


def attack(bkz_block_size: int, max_columns: int | None) -> None:
    """Przeprowadza atak Primal na FrodoKEM."""
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie klucza publicznego (do ataku)...", end="")
    frodo_public = load_frodo(False)
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")
    
    # Wczytaj też klucz prywatny dla weryfikacji (opcjonalne)
    with Printer(Style.LOG) as printer:
        printer("Wczytywanie klucza prywatnego (do weryfikacji)...", end="")
    try:
        frodo_private = load_frodo(True)
        frodo_public._S = frodo_private._S
        frodo_public._E = frodo_private._E
        with Printer(Style.LOG) as printer:
            printer(" Wykonano")
    except SystemExit:
        with Printer(Style.LOG) as printer:
            printer(" Pominięto (brak klucza prywatnego)")
    
    with Printer(Style.INFO) as printer:
        printer(f"Parametry ataku:")
        printer(f"  Rozmiar bloku BKZ: {bkz_block_size}")
        if max_columns:
            printer(f"  Maksymalna liczba kolumn: {max_columns}")
    
    # Przeprowadź atak
    recovered_S, attack_time = primal_attack_frodo(
        frodo_public, 
        block_size=bkz_block_size,
        max_columns=max_columns
    )
    
    if recovered_S is None:
        with Printer(Style.FAIL) as printer:
            printer("\nAtak nie powiódł się.")
        return
    
    with Printer(Style.SUCCESS) as printer:
        printer(f"\nOdzyskano sekret w czasie {attack_time:.2f}s")
    
    # Spróbuj dekapsulacji z odzyskanym sekretem
    with Printer(Style.LOG) as printer:
        printer("\nWczytywanie szyfrogramu...", end="")
    try:
        ciphertext, original_shared_secret = load_ciphertext()
        with Printer(Style.LOG) as printer:
            printer(" Wykonano")
        
        # Podstaw odzyskany sekret
        frodo_public._S = recovered_S
        
        with Printer(Style.LOG) as printer:
            printer("Dekapsulacja z odzyskanym sekretem...", end="")
        recovered_secret = frodo_public.decaps(ciphertext)
        with Printer(Style.LOG) as printer:
            printer(" Wykonano")
        
        with Printer(Style.INFO) as printer:
            printer(f"Oryginalny klucz:  {original_shared_secret.hex()}")
            printer(f"Odzyskany klucz:   {recovered_secret.hex()}")
        
        if recovered_secret == original_shared_secret:
            with Printer(Style.SUCCESS) as printer:
                printer("\n" + "=" * 50)
                printer("ATAK ZAKOŃCZONY PEŁNYM SUKCESEM!")
                printer("Odzyskano klucz współdzielony!")
                printer("=" * 50)
        else:
            with Printer(Style.FAIL) as printer:
                printer("\nKlucze nie są zgodne.")
    
    except SystemExit:
        with Printer(Style.LOG) as printer:
            printer(" Pominięto (brak szyfrogramu)")


def demo() -> None:
    """Pełna demonstracja: generowanie, enkapsulacja, dekapsulacja i atak."""
    with Printer(Style.INFO) as printer:
        printer("=" * 60)
        printer("        PEŁNA DEMONSTRACJA ATAKU PRIMAL NA FRODOKEM")
        printer("=" * 60)
        printer("")
        printer("Ten skrypt demonstruje:")
        printer("  1. Generowanie kluczy FrodoKEM")
        printer("  2. Enkapsulację klucza współdzielonego")
        printer("  3. Legalną dekapsulację (z kluczem prywatnym)")
        printer("  4. Atak Primal (odzyskanie sekretu z klucza publicznego)")
        printer("  5. Dekapsulację przy użyciu odzyskanego sekretu")
        printer("")
    
    # Krok 1: Generowanie kluczy
    with Printer(Style.INFO) as printer:
        printer("-" * 60)
        printer("KROK 1: Generowanie kluczy")
        printer("-" * 60)
    
    frodo = generate("demo")
    
    # Krok 2: Enkapsulacja
    with Printer(Style.INFO) as printer:
        printer("")
        printer("-" * 60)
        printer("KROK 2: Enkapsulacja klucza")
        printer("-" * 60)
    
    encapsulate()
    
    # Krok 3: Legalna dekapsulacja
    with Printer(Style.INFO) as printer:
        printer("")
        printer("-" * 60)
        printer("KROK 3: Legalna dekapsulacja (z kluczem prywatnym)")
        printer("-" * 60)
    
    decapsulate()
    
    # Krok 4: Atak
    with Printer(Style.INFO) as printer:
        printer("")
        printer("-" * 60)
        printer("KROK 4: Atak Primal")
        printer("-" * 60)
    
    attack(bkz_block_size=20, max_columns=None)


def get_param(
    index: int,
    type: typing.Type[int | float | str],
    name: str,
    default: int | float | str | None,
) -> int | float | str | None:
    """Pobiera parametr z linii poleceń."""
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


def print_help() -> None:
    """Wyświetla pomoc."""
    with Printer(Style.INFO) as printer:
        # cspell: disable
        printer("Dostępne akcje:")
        printer("  p[omoc] - wyświetla tę pomoc")
        printer("")
        printer("  g[eneruj] [parametry] - generuje nowe klucze FrodoKEM")
        printer("    Parametry: demo (domyślne), toy, frodo640")
        printer("    Przykład: python main_frodo.py g demo")
        printer("")
        printer("  e[nkapsuluj] - enkapsuluje klucz współdzielony")
        printer("    Wymaga wcześniejszego wygenerowania kluczy.")
        printer("")
        printer("  d[ekapsuluj] - dekapsuluje klucz współdzielony")
        printer("    Wymaga klucza prywatnego i szyfrogramu.")
        printer("")
        printer("  a[takuj] [rozmiar_bloku] [max_kolumn] - przeprowadza atak Primal")
        printer("    Domyślny rozmiar bloku BKZ: 20")
        printer("    Domyślna maksymalna liczba kolumn: wszystkie")
        printer("    Przykład: python main_frodo.py a 25 2")
        printer("")
        printer("  demo - pełna demonstracja (generowanie, enkapsulacja, atak)")
        printer("")
        printer("Zestawy parametrów:")
        printer("  demo     - n=10, m=60, q=251, bardzo słabe (szybki atak)")
        printer("  toy      - n=15, m=80, q=251, słabe")
        printer("  frodo640 - n=32, m=128, q=32749, zredukowane parametry")
        printer("")
        printer("Przykłady użycia:")
        printer("  python main_frodo.py demo              # Pełna demonstracja")
        printer("  python main_frodo.py g toy             # Generuj klucze 'toy'")
        printer("  python main_frodo.py e                 # Enkapsuluj")
        printer("  python main_frodo.py a 25              # Atak z blokiem BKZ-25")
        # cspell: enable


def main():
    """Główna funkcja programu."""
    with Printer(Style.BANNER) as printer:
        printer("=" * 50)
        printer("        ATAK PRIMAL NA FRODOKEM")
        printer("   Mechanizm Enkapsulacji Klucza (KEM)")
        printer("       oparty na problemie LWE")
        printer("=" * 50)
    
    action = get_param(1, str, "akcja", default=None)
    
    if action is None:
        with Printer(Style.ERROR) as printer:
            printer("Akcja jest wymagana")
            printer("Użyj 'python main_frodo.py pomoc' aby wyświetlić dostępne akcje")
        sys.exit(1)
    
    if "pomoc".startswith(action) or action in ["-h", "--help"]:
        print_help()
    
    elif "generuj".startswith(action):
        params_name = get_param(2, str, "parametry", default="demo")
        generate(params_name)
    
    elif "enkapsuluj".startswith(action):
        encapsulate()
    
    elif "dekapsuluj".startswith(action):
        decapsulate()
    
    elif "atakuj".startswith(action):
        bkz_block_size = get_param(2, int, "rozmiar bloku BKZ", default=20)
        max_columns_str = get_param(3, str, "maksymalna liczba kolumn", default=None)
        max_columns = int(max_columns_str) if max_columns_str else None
        attack(bkz_block_size, max_columns)
    
    elif action == "demo":
        demo()
    
    else:
        with Printer(Style.ERROR) as printer:
            printer(f"Nieznana akcja: {action}")
            printer("Użyj 'python main_frodo.py pomoc' aby wyświetlić dostępne akcje")
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
        import traceback
        traceback.print_exc()
        sys.exit(1)
