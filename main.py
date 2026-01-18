#!/usr/bin/env python3

import sys
import typing
from src.printer import Printer, Style
from src.lwe import LWE
from src.attack import primal_attack
from src.regev import encrypt, decrypt


def get_param(
    index: int, name: str, type: typing.Type[int | float], default: int | float
) -> int | float:
    if len(sys.argv) > index:
        try:
            if type == int:
                return int(sys.argv[index])
            elif type == float:
                return float(sys.argv[index])
            else:
                raise ValueError(f"Niepoprawny typ: {type}")
        except Exception as e:
            with Printer(Style.ERROR) as printer:
                printer(f"Niepoprawna wartość parametru {name}: {e}")
            sys.exit(1)

    with Printer(Style.WARNING) as printer:
        printer(f"Przyjęto domyślną wartość parametru {name}: {default}")
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


if __name__ == "__main__":
    with Printer(Style.BANNER) as printer:
        printer("=========================================")
        printer("       PROJEKT: ATAK PRIMAL NA LWE       ")
        printer("=========================================")

    message = get_param(1, "message", int, default=None)
    if message is None:
        with Printer(Style.ERROR) as printer:
            printer("Wiadomość jest wymagana")
        sys.exit(1)
    elif message not in [0, 1]:
        with Printer(Style.ERROR) as printer:
            printer(f"Wiadomość musi być 0 lub 1, otrzymano: {message}")
        sys.exit(1)

    n = get_param(2, "n", int, default=10)
    m = get_param(3, "m", int, default=60)
    q = get_param(4, "q", int, default=101)

    if not is_prime(q):
        with Printer(Style.ERROR) as printer:
            printer(f"Parametr q musi być liczbą pierwszą, otrzymano: {q}")
        sys.exit(1)

    alpha = get_param(5, "alpha", float, default=0.01)

    bkz_block_size = get_param(6, "bkz_block_size", int, default=30)

    with Printer(Style.LOG) as printer:
        printer(
            f"Wybrane parametry: n={n}, m={m}, q={q}, alpha={alpha}, bkz_block_size={bkz_block_size}"
        )
        printer(f"Generowanie instancji LWE...", end="")
    lwe = LWE(n, m, q, alpha)
    A, b = lwe.generate()
    with Printer(Style.LOG) as printer:
        printer(" Wykonano")

    ciphertext = encrypt(lwe, message)
    decrypted_message = decrypt(lwe, ciphertext)
    with Printer(Style.INFO) as printer:
        printer(f"Wiadomość zaszyfrowana: {ciphertext}")
        printer(f"Wiadomość odszyfrowana: {decrypted_message}")
        printer(f"Przygotowania zakończone, przechodzimy do ataku.")

    recovered_s, time_taken = primal_attack(lwe, bkz_block_size)

    with Printer(Style.INFO) as printer:
        printer(f"Atak zakończony w {time_taken:.2f}s")

    with Printer(Style.BANNER) as printer:
        printer("=========================================")
        printer("                WYNIKI                   ")
        printer("=========================================")

    if recovered_s is not None:
        with Printer(Style.SUCCESS) as printer:
            printer(
                f"Znaleziono {"dokładny" if lwe.check_solution_exact(recovered_s) else "równoważny"} sekret!"
            )
            printer(f"Oryginalny s:   {lwe._s}")
            printer(f"Odzyskany s:    {recovered_s}")
    else:
        with Printer(Style.FAIL) as printer:
            printer(f"Nie udało się odzyskać sekretu.")
            printer("Sugestie:")
            printer(" - Zwiększ parametr block_size w BKZ")
            printer(" - Zwiększ liczbę próbek m")
            printer(" - Zmniejsz szum (alpha)")

    with Printer(Style.BANNER) as printer:
        printer("=========================================")
