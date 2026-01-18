# Primal

Implementacja ataku typu *primal* na przykładzie kryptosystemu klucza publicznego zaproponowanego przez Regev'a w 2005 opartego na problemie *Learning with errors*.

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

Program może pracować w jednym z 5 trybów, wybieranym przez pierwszy argument:

```sh
./main.py pomoc
./main.py generuj [n] [m] [q] [alpha]
./main.py szyfruj <wiadomość>
./main.py odszyfruj
./main.py atakuj [rozmiar bloku BKZ]
```

Wywołaj tryb `pomoc`, aby dowiedzieć się więcej.

Program honoruje także flagi w postaci zmiennych środowiskowych o wartościach `"0"`/`"1"`:
- PRIMAL_BANNER - kontroluje wyświetlanie banerów, domyślnie `"1"`
- PRIMAL_LOG - kontroluje wyświetlanie szczegółowych logów, domyślnie `"1"`
- PRIMAL_COLORS - kontroluje wykorzystanie kolorów w konsoli, domyślnie `"1"`
