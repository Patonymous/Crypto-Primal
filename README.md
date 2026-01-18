# Primal

Implementacja ataku typu *primal* na kryptosystemy oparte na problemie LWE.

Wykonana w ramach projektu na przedmiot Wprowadzenie do współczesnej kryptologii
na wydziale Matematyki i Nauk Informacyjnych Politechniki Warszawskiej.

# Instrukcja obsługi programu

Wymaga python 3.

## Przygotowanie środowiska

1. Opcjonalne, ale zalecane - stworzenie i aktywacja python virtual environment
2. Instalacja zależności

```sh
python -m pip install -r requirements.txt
```

> **ℹ️ UWAGA:**  
> Jeśli napotkasz problemy z instalacją `fpylll` dotyczące pliku `gmp.h`
> lub innych związanych z biblioteką GMP prawdopodobnie nie masz jej zainstalowanej.
> Na Linux pobierz ją za pomocą swojego package manager'a.
> Na Windows spróbuj wykorzystać środowisko `conda` do instalacji lub pracuj w WSL-u.

## Uruchomienie programu

Program przyjmuje 5 opcjonalnych parametrów pozycyjnych:

```sh
./main.py bit [n] [m] [q] [alpha] [block_size]
```

gdzie:
- bit - wiadomość do zaszyfrowania, musi wynosić 0 albo 1
- n - rozmiar sekretu, zwiększenie zmniejsza szansę powodzenia ataku
- m - liczba próbek, zwiększenie zwiększa szansę powodzenia ataku, ale też czas trwania
- q - moduł, powinien być liczbą pierwszą, zwiększenie zmniejsza szansę powodzenia ataku
- alpha - poziom szumu, zwiększenie zmniejsza szansę powodzenia ataku
- block_size - siła redukcji kraty, zwiększenie zwiększa szansę powodzenia ataku, ale też czas trwania

Program honoruje także flagi w postaci zmiennych środowiskowych o wartościach `"0"`/`"1"`:
- PRIMAL_BANNER - kontroluje wyświetlanie banerów, domyślnie `"1"`
- PRIMAL_LOG - kontroluje wyświetlanie szczegółowych logów, domyślnie `"1"`
- PRIMAL_COLORS - kontroluje wykorzystanie kolorów w konsoli, domyślnie `"1"`
