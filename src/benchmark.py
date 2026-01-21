import numpy as np
import time
from dataclasses import dataclass
from src.lwe import LWE
from src.attack import primal_attack
from src.printer import Printer, Style


@dataclass
class BenchmarkResult:
    """Wynik pojedynczego testu."""
    block_size: int
    time_seconds: float
    success: bool


def run_single_benchmark(
    n: int, m: int, q: int, alpha: float, block_size: int
) -> BenchmarkResult:
    """Uruchamia pojedynczy test benchmarkowy."""
    lwe = LWE()
    lwe.generate(n, m, q, alpha)
    
    start = time.time()
    found_s, _ = primal_attack(lwe, block_size)
    elapsed = time.time() - start
    
    return BenchmarkResult(
        block_size=block_size,
        time_seconds=elapsed,
        success=(found_s is not None)
    )


def compare_block_sizes(
    n: int = 10, m: int = 60, q: int = 101, alpha: float = 0.01,
    block_sizes: list[int] = None, trials: int = 3
) -> dict:
    """
    Porównuje różne rozmiary bloków BKZ.
    
    Args:
        n, m, q, alpha: Parametry LWE
        block_sizes: Lista rozmiarów bloków do przetestowania
        trials: Liczba powtórzeń dla każdego rozmiaru
        
    Returns:
        Słownik z wynikami dla każdego rozmiaru bloku
    """
    if block_sizes is None:
        block_sizes = [10, 15, 20, 25, 30]
    
    results = {}
    
    with Printer(Style.INFO) as p:
        p(f"Porównanie rozmiarów bloków BKZ (n={n}, m={m}, q={q}, α={alpha})")
        p(f"Bloków: {block_sizes}, prób: {trials}")
        p("-" * 60)
    
    for bs in block_sizes:
        times = []
        successes = 0
        
        for trial in range(trials):
            with Printer(Style.LOG) as p:
                p(f"BKZ-{bs}, próba {trial+1}/{trials}...", end="")
            
            result = run_single_benchmark(n, m, q, alpha, bs)
            times.append(result.time_seconds)
            if result.success:
                successes += 1
            
            with Printer(Style.LOG) as p:
                p(f" {result.time_seconds:.2f}s, {'✓' if result.success else '✗'}")
        
        results[bs] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'success_rate': successes / trials,
            'trials': trials
        }
        
        with Printer(Style.INFO) as p:
            sr = results[bs]['success_rate']
            at = results[bs]['avg_time']
            p(f"BKZ-{bs}: sukces={sr:.0%}, czas={at:.2f}s ± {results[bs]['std_time']:.2f}s")
    
    return results


def print_comparison_table(results: dict):
    """Wyświetla tabelę porównawczą rozmiarów bloków."""
    with Printer(Style.INFO) as p:
        p("\n" + "=" * 50)
        p("PODSUMOWANIE PORÓWNANIA")
        p("=" * 50)
        p(f"{'Blok':<10} {'Sukces':<12} {'Czas śr.':<12} {'Odch. std.':<12}")
        p("-" * 50)
    
    for bs, data in sorted(results.items()):
        with Printer(Style.INFO) as p:
            p(f"BKZ-{bs:<6} {data['success_rate']:>6.0%}      "
              f"{data['avg_time']:>8.2f}s    {data['std_time']:>8.2f}s")
    
    with Printer(Style.INFO) as p:
        p("=" * 50)


@dataclass
class ParamBenchmarkResult:
    """Wynik testu dla konkretnych parametrów."""
    n: int
    m: int
    q: int
    alpha: float
    block_size: int
    time_seconds: float
    success: bool


def compare_parameters(
    param_sets: list[dict], block_size: int = 25, trials: int = 3
) -> list[dict]:
    """
    Porównuje różne zestawy parametrów LWE.
    
    Args:
        param_sets: Lista słowników z parametrami {n, m, q, alpha}
        block_size: Rozmiar bloku BKZ do użycia
        trials: Liczba powtórzeń dla każdego zestawu
        
    Returns:
        Lista słowników z wynikami dla każdego zestawu parametrów
    """
    results = []
    
    with Printer(Style.INFO) as p:
        p(f"Porównanie parametrów LWE (BKZ-{block_size}, prób: {trials})")
        p("-" * 70)
    
    for i, params in enumerate(param_sets):
        n = params.get('n', 10)
        m = params.get('m', 60)
        q = params.get('q', 101)
        alpha = params.get('alpha', 0.01)
        
        with Printer(Style.INFO) as p:
            p(f"\nZestaw {i+1}: n={n}, m={m}, q={q}, α={alpha}")
        
        times = []
        successes = 0
        
        for trial in range(trials):
            with Printer(Style.LOG) as p:
                p(f"  Próba {trial+1}/{trials}...", end="")
            
            result = run_single_benchmark(n, m, q, alpha, block_size)
            times.append(result.time_seconds)
            if result.success:
                successes += 1
            
            with Printer(Style.LOG) as p:
                p(f" {result.time_seconds:.2f}s, {'✓' if result.success else '✗'}")
        
        result_data = {
            'n': n, 'm': m, 'q': q, 'alpha': alpha,
            'block_size': block_size,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'success_rate': successes / trials,
            'trials': trials
        }
        results.append(result_data)
        
        with Printer(Style.INFO) as p:
            sr = result_data['success_rate']
            at = result_data['avg_time']
            p(f"  Wynik: sukces={sr:.0%}, czas={at:.2f}s ± {result_data['std_time']:.2f}s")
    
    return results


def print_params_comparison_table(results: list[dict]):
    """Wyświetla tabelę porównawczą parametrów."""
    with Printer(Style.INFO) as p:
        p("\n" + "=" * 75)
        p("PODSUMOWANIE PORÓWNANIA PARAMETRÓW")
        p("=" * 75)
        p(f"{'n':<5} {'m':<5} {'q':<8} {'alpha':<8} {'Sukces':<10} {'Czas śr.':<12}")
        p("-" * 75)
    
    for data in results:
        with Printer(Style.INFO) as p:
            p(f"{data['n']:<5} {data['m']:<5} {data['q']:<8} {data['alpha']:<8.3f} "
              f"{data['success_rate']:>6.0%}     {data['avg_time']:>8.2f}s")
    
    with Printer(Style.INFO) as p:
        p("=" * 75)
