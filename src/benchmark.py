import numpy as np
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from src.lwe import LWE
from src.attack import primal_attack, build_primal_lattice
from src.printer import Printer, Style
from fpylll import IntegerMatrix, LLL, BKZ, GSO


def set_random_seed(seed: int | None = None) -> int:
    """Ustawia ziarno losowości dla powtarzalności eksperymentów."""
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
    np.random.seed(seed)
    return seed


def _is_interactive():
    """Sprawdza czy backend matplotlib jest interaktywny."""
    import matplotlib
    return matplotlib.is_interactive() or matplotlib.get_backend() not in ['agg', 'Agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg']


def compute_basis_norms(B: IntegerMatrix) -> list[float]:
    """Oblicza normy wektorów w bazie."""
    return [np.linalg.norm([B[i, j] for j in range(B.ncols)]) for i in range(B.nrows)]


def compute_gs_norms(B: IntegerMatrix) -> list[float]:
    """Oblicza normy wektorów Gram-Schmidta."""
    M = GSO.Mat(B)
    M.update_gso()
    return [M.get_r(i, i) ** 0.5 for i in range(B.nrows)]


@dataclass
class BenchmarkResult:
    """Wynik pojedynczego testu."""
    block_size: int
    time_seconds: float
    success: bool
    lattice_dim: int = 0  # Wymiar kraty dla diagnostyki


def run_single_benchmark(
    n: int, m: int, q: int, alpha: float, block_size: int
) -> BenchmarkResult:
    """Uruchamia pojedynczy test benchmarkowy."""
    lwe = LWE()
    lwe.generate(n, m, q, alpha)
    
    start = time.time()
    found_s, _ = primal_attack(lwe, block_size)
    elapsed = time.time() - start
    
    # Weryfikuj dokładnie czy znaleziony sekret jest poprawny
    success = lwe.check_solution_exact(found_s)
    
    return BenchmarkResult(
        block_size=block_size,
        time_seconds=elapsed,
        success=success,
        lattice_dim=m + n + 1
    )


def compare_block_sizes(
    n: int = 10, m: int = 60, q: int = 101, alpha: float = 0.01,
    block_sizes: list[int] = None, trials: int = 5, seed: int | None = None
) -> dict:
    """
    Porównuje różne rozmiary bloków BKZ.
    
    Args:
        n, m, q, alpha: Parametry LWE
        block_sizes: Lista rozmiarów bloków do przetestowania
        trials: Liczba powtórzeń dla każdego rozmiaru (zalecane >= 5)
        seed: Ziarno losowości dla powtarzalności
        
    Returns:
        Słownik z wynikami dla każdego rozmiaru bloku
    """
    if block_sizes is None:
        block_sizes = [10, 15, 20, 25, 30]
    
    used_seed = set_random_seed(seed)
    results = {}
    
    with Printer(Style.INFO) as p:
        p(f"Porównanie rozmiarów bloków BKZ (n={n}, m={m}, q={q}, α={alpha})")
        p(f"Bloków: {block_sizes}, prób: {trials}, seed: {used_seed}")
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
    param_sets: list[dict], block_size: int = 25, trials: int = 5, seed: int | None = None
) -> list[dict]:
    """
    Porównuje różne zestawy parametrów LWE.
    
    Args:
        param_sets: Lista słowników z parametrami {n, m, q, alpha}
        block_size: Rozmiar bloku BKZ do użycia
        trials: Liczba powtórzeń dla każdego zestawu (zalecane >= 5)
        seed: Ziarno losowości dla powtarzalności
        
    Returns:
        Lista słowników z wynikami dla każdego zestawu parametrów
    """
    used_seed = set_random_seed(seed)
    results = []
    
    with Printer(Style.INFO) as p:
        p(f"Porównanie parametrów LWE (BKZ-{block_size}, prób: {trials}, seed: {used_seed})")
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


def plot_block_size_comparison(results: dict, save_path: str = None):
    """
    Tworzy wykres porównawczy rozmiarów bloków BKZ.
    
    Args:
        results: Słownik z wynikami z compare_block_sizes()
        save_path: Ścieżka do zapisu wykresu
    """
    block_sizes = sorted(results.keys())
    success_rates = [results[bs]['success_rate'] * 100 for bs in block_sizes]
    avg_times = [results[bs]['avg_time'] for bs in block_sizes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wykres 1: Współczynnik sukcesu
    ax1.plot(block_sizes, success_rates, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Rozmiar bloku BKZ', fontsize=11)
    ax1.set_ylabel('Wskaźnik sukcesu ataku (%)', fontsize=11)
    ax1.set_title('Skuteczność ataku w zależności od rozmiaru bloku BKZ', fontsize=12)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    
    # Wykres 2: Czas wykonania
    ax2.plot(block_sizes, avg_times, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Rozmiar bloku BKZ', fontsize=11)
    ax2.set_ylabel('Średni czas wykonania (s)', fontsize=11)
    ax2.set_title('Czas ataku w zależności od rozmiaru bloku BKZ', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano: {save_path}")
        plt.close()
    
    return fig


def save_results_to_json(results: list[dict], save_path: str) -> None:
    """
    Zapisuje wyniki benchmarku do pliku JSON.
    
    Args:
        results: Lista słowników z wynikami
        save_path: Ścieżka do zapisu pliku JSON
    """
    import json
    from pathlib import Path
    
    # Upewnij się że katalog istnieje
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Przygotuj dane w czytelnym formacie
    formatted_results = []
    for i, r in enumerate(results, 1):
        formatted_results.append({
            'set_id': i,
            'parameters': {
                'n': r['n'],
                'm': r['m'],
                'q': r['q'],
                'alpha': r['alpha']
            },
            'results': {
                'block_size': r['block_size'],
                'success_rate': r['success_rate'],
                'avg_time_seconds': round(r['avg_time'], 3),
                'std_time_seconds': round(r['std_time'], 3),
                'trials': r['trials']
            }
        })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, indent=2, ensure_ascii=False)
    
    print(f"Zapisano wyniki do: {save_path}")


def plot_heatmap(results: list[dict], x_param: str, y_param: str, 
                 metric: str = 'success_rate', save_path: str = None):
    """
    Tworzy uniwersalną heatmapę dla różnych parametrów.
    
    Args:
        results: Lista słowników z wynikami
        x_param: Parametr dla osi X ('n', 'm', 'q', 'alpha')
        y_param: Parametr dla osi Y ('n', 'm', 'q', 'alpha')
        metric: Metryka do wyświetlenia ('success_rate' lub 'avg_time')
        save_path: Ścieżka do zapisu wykresu
    """
    # Mapowanie nazw parametrów na wyświetlane etykiety
    param_labels = {
        'n': 'Wymiar sekretu (n)',
        'm': 'Liczba próbek (m)',
        'q': 'Moduł (q)',
        'alpha': 'Parametr szumu (α)'
    }
    
    metric_labels = {
        'success_rate': 'Współczynnik sukcesu',
        'avg_time': 'Średni czas (s)'
    }
    
    # Zbierz unikalne wartości
    x_vals = sorted(set(r[x_param] for r in results))
    y_vals = sorted(set(r[y_param] for r in results))
    
    if len(x_vals) < 2 or len(y_vals) < 2:
        print(f"Zbyt mało różnych wartości {x_param} lub {y_param} do stworzenia heatmapy")
        return None
    
    # Utwórz macierz
    matrix = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in results:
        try:
            i = y_vals.index(r[y_param])
            j = x_vals.index(r[x_param])
            matrix[i, j] = r[metric]
        except (ValueError, KeyError):
            continue
    
    # Sprawdź czy są jakieś dane
    if np.all(np.isnan(matrix)):
        print(f"Brak danych dla heatmapy {y_param} vs {x_param}")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Wybór mapy kolorów i zakresu w zależności od metryki
    if metric == 'success_rate':
        cmap = 'RdYlGn'
        vmin, vmax = 0, 1
        fmt = '.0%'
    else:  # avg_time
        cmap = 'YlOrRd_r'  # Odwrócone - krótszy czas = lepiej
        vmin, vmax = None, None
        fmt = '.2f'
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    ax.set_xticks(range(len(x_vals)))
    ax.set_yticks(range(len(y_vals)))
    ax.set_xticklabels([f'{v:.3f}' if isinstance(v, float) else str(v) for v in x_vals])
    ax.set_yticklabels([f'{v:.3f}' if isinstance(v, float) else str(v) for v in y_vals])
    ax.set_xlabel(param_labels.get(x_param, x_param), fontsize=11)
    ax.set_ylabel(param_labels.get(y_param, y_param), fontsize=11)
    ax.set_title(f'{metric_labels.get(metric, metric)} ataku Primal', fontsize=12)
    
    # Dodaj wartości tekstowe
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                if metric == 'success_rate':
                    text_val = f'{val:.0%}'
                    text_color = 'white' if val < 0.5 else 'black'
                else:
                    text_val = f'{val:.2f}'
                    text_color = 'black'
                
                ax.text(j, i, text_val, ha='center', va='center', 
                       fontsize=10, color=text_color)
    
    plt.colorbar(im, ax=ax, label=metric_labels.get(metric, metric))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano: {save_path}")
    
    return fig


def plot_success_heatmap_from_params(results: list[dict], save_path: str = None):
    """
    Tworzy heatmapę sukcesu w funkcji wymiaru n i parametru alpha (wrapper dla kompatybilności).
    """
    return plot_heatmap(results, 'n', 'alpha', 'success_rate', save_path)


def plot_parameter_impact(results: list[dict], param_name: str, output_dir: str = ".") -> list[str]:
    """
    Generuje wykres pokazujący wpływ parametru na sukces ataku i czas wykonania.
    Uwzględnia tylko wyniki gdzie POZOSTAŁE parametry są stałe (najbardziej popularna kombinacja).
    
    Args:
        results: Lista słowników z wynikami
        param_name: Nazwa parametru ('n', 'm', 'q', 'alpha')
        output_dir: Katalog do zapisu wykresów
        
    Returns:
        Lista ścieżek do wygenerowanych plików
    """
    import numpy as np
    from pathlib import Path
    from collections import Counter
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Znajdź najbardziej popularną kombinację pozostałych parametrów
    other_params = ['n', 'm', 'q', 'alpha']
    other_params.remove(param_name)
    
    # Zlicz kombinacje pozostałych parametrów
    combinations = [tuple(r[p] for p in other_params) for r in results]
    most_common_combo = Counter(combinations).most_common(1)[0][0]
    
    # Filtruj wyniki tylko do tej kombinacji
    filtered_results = [
        r for r in results 
        if tuple(r[p] for p in other_params) == most_common_combo
    ]
    
    if len(filtered_results) < 2:
        return []
    
    # Grupuj wyniki według wartości parametru
    param_values = sorted(set(r[param_name] for r in filtered_results))
    
    if len(param_values) < 2:
        return []
    
    success_rates = []
    avg_times = []
    
    for val in param_values:
        matching = [r for r in filtered_results if r[param_name] == val]
        # Każda wartość powinna mieć dokładnie 1 wynik (bo już przefiltrowane)
        success_rates.append(np.mean([r['success_rate'] for r in matching]) * 100)  # Konwersja do %
        avg_times.append(np.mean([r['avg_time'] for r in matching]))
    
    param_labels = {
        'n': ('Wymiar problemu LWE (n)', 'Wymiar n'),
        'm': ('Liczba próbek LWE (m)', 'Liczba próbek m'),
        'q': ('Moduł (q)', 'Moduł q'),
        'alpha': ('Parametr szumu (α)', 'Parametr α')
    }
    
    full_label, short_label = param_labels.get(param_name, (param_name, param_name))
    
    # Utwórz wykres z dwoma panelami obok siebie
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wykres 1: Wpływ na sukces ataku
    ax1.plot(param_values, success_rates, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel(full_label, fontsize=11)
    ax1.set_ylabel('Wskaźnik sukcesu ataku (%)', fontsize=11)
    ax1.set_title(f'Skuteczność ataku w zależności od {short_label}', fontsize=12)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    
    # Wykres 2: Wpływ na czas wykonania
    ax2.plot(param_values, avg_times, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel(full_label, fontsize=11)
    ax2.set_ylabel('Średni czas wykonania (s)', fontsize=11)
    ax2.set_title(f'Czas ataku w zależności od {short_label}', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    combined_path = f"{output_dir}/param_{param_name}.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return [combined_path]


def generate_parameter_plots(results: list[dict], output_dir: str = ".") -> list[str]:
    """
    Generuje wykresy badające wpływ wszystkich parametrów.
    
    Args:
        results: Lista słowników z wynikami
        output_dir: Katalog do zapisu wykresów
        
    Returns:
        Lista ścieżek do wygenerowanych plików
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_files = []
    
    # Generuj wykresy dla każdego parametru
    for param in ['n', 'alpha', 'm', 'q']:
        files = plot_parameter_impact(results, param, output_dir)
        all_files.extend(files)
    
    return all_files


def generate_key_heatmaps(results: list[dict], output_dir: str = ".") -> list[str]:
    """
    Generuje 3 kluczowe heatmapy pokazujące interakcje między parametrami.
    
    Args:
        results: Lista słowników z wynikami
        output_dir: Katalog do zapisu wykresów
        
    Returns:
        Lista ścieżek do wygenerowanych plików
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # 3 najważniejsze heatmapy dla analizy LWE:
    heatmap_configs = [
        ('n', 'alpha', 'success_rate', 'n_vs_alpha_success'),  # Kluczowa: wymiar vs szum -> sukces
        ('n', 'alpha', 'avg_time', 'n_vs_alpha_time'),          # Wymiar vs szum -> czas
        ('n', 'm', 'success_rate', 'n_vs_m_success'),           # Wymiar vs próbki -> sukces
    ]
    
    for x_param, y_param, metric, filename in heatmap_configs:
        # Sprawdź czy są wystarczające dane
        x_vals = set(r[x_param] for r in results)
        y_vals = set(r[y_param] for r in results)
        
        if len(x_vals) >= 2 and len(y_vals) >= 2:
            save_path = f"{output_dir}/heatmap_{filename}.png"
            fig = plot_heatmap(results, x_param, y_param, metric, save_path)
            if fig is not None:
                generated_files.append(save_path)
    
    return generated_files


def visualize_reduction(lwe: LWE, block_sizes: list[int] = None, save_path: str = None):
    """
    Wizualizuje jakość redukcji dla różnych algorytmów.
    
    Args:
        lwe: Instancja LWE
        block_sizes: Lista rozmiarów bloków BKZ (domyślnie [10, 20, 30])
        save_path: Ścieżka do zapisu wykresu
    """
    if block_sizes is None:
        block_sizes = [10, 20, 30]
    
    B_orig = build_primal_lattice(lwe)
    norms_orig = compute_basis_norms(B_orig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['b', 'g', 'r', 'm', 'c']
    
    # Wykres 1: Normy wektorów bazy
    ax1.semilogy(norms_orig, 'k--', label='Oryginalna', linewidth=2, alpha=0.5)
    
    B_lll = B_orig.__copy__()
    LLL.reduction(B_lll)
    ax1.semilogy(compute_basis_norms(B_lll), 'b-', label='LLL', linewidth=2)
    
    for i, bs in enumerate(block_sizes):
        B_bkz = B_lll.__copy__()
        BKZ.reduction(B_bkz, BKZ.Param(bs))
        ax1.semilogy(compute_basis_norms(B_bkz), f'{colors[(i+1) % len(colors)]}-', 
                    label=f'BKZ-{bs}', linewidth=2)
    
    ax1.set_xlabel('Indeks wektora')
    ax1.set_ylabel('Norma (log)')
    ax1.set_title('Normy wektorów bazy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Wykres 2: Profil Gram-Schmidta
    ax2.semilogy(compute_gs_norms(B_orig), 'k--', label='Oryginalna', linewidth=2, alpha=0.5)
    ax2.semilogy(compute_gs_norms(B_lll), 'b-', label='LLL', linewidth=2)
    
    for i, bs in enumerate(block_sizes):
        B_bkz = B_lll.__copy__()
        BKZ.reduction(B_bkz, BKZ.Param(bs))
        ax2.semilogy(compute_gs_norms(B_bkz), f'{colors[(i+1) % len(colors)]}-', 
                    label=f'BKZ-{bs}', linewidth=2)
    
    ax2.set_xlabel('Indeks wektora')
    ax2.set_ylabel('Norma GS (log)')
    ax2.set_title('Profil Gram-Schmidta')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano: {save_path}")
    
    if _is_interactive():
        plt.show()
    return fig


def visualize_block_size_impact(lwe: LWE, max_block: int = 40, save_path: str = None):
    """
    Pokazuje jak norma pierwszego wektora zmienia się z rozmiarem bloku BKZ.
    
    Args:
        lwe: Instancja LWE
        max_block: Maksymalny rozmiar bloku BKZ
        save_path: Ścieżka do zapisu wykresu
    """
    B = build_primal_lattice(lwe)
    
    # Norma przed redukcją (oryginalna baza)
    orig_norm = compute_basis_norms(B)[0]
    
    # Redukcja LLL
    LLL.reduction(B)
    lll_norm = compute_basis_norms(B)[0]
    
    block_sizes = list(range(5, min(max_block, B.nrows // 2), 5))
    first_norms = []
    
    for bs in block_sizes:
        B_copy = B.__copy__()
        BKZ.reduction(B_copy, BKZ.Param(bs))
        first_norms.append(compute_basis_norms(B_copy)[0])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Linia LLL jako odniesienie
    ax.axhline(y=lll_norm, color='orange', linestyle='--', linewidth=2, 
               label=f'LLL (norma={lll_norm:.1f})', alpha=0.8)
    
    # Główna krzywa BKZ
    ax.plot(block_sizes, first_norms, 'bo-', linewidth=2, markersize=8, label='BKZ')
    
    ax.set_xlabel('Rozmiar bloku BKZ')
    ax.set_ylabel('Długość najkrótszego wektora w bazie (||b₁||)')
    ax.set_title('Wpływ rozmiaru bloku BKZ na jakość redukcji')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Zaznacz minimum
    min_idx = np.argmin(first_norms)
    improvement = (lll_norm - first_norms[min_idx]) / lll_norm * 100
    ax.annotate(f'Min: {first_norms[min_idx]:.1f}\n(BKZ-{block_sizes[min_idx]})\n↓{improvement:.1f}% vs LLL',
               xy=(block_sizes[min_idx], first_norms[min_idx]),
               xytext=(10, 20), textcoords='offset points',
               bbox=dict(boxstyle='round', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano: {save_path}")
    
    if _is_interactive():
        plt.show()
    return fig
