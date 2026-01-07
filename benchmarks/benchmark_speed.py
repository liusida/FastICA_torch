"""
Speed benchmarks comparing fastica_torch to sklearn's FastICA.

This script measures execution time for various data sizes and configurations,
comparing PyTorch (CPU) implementation against sklearn.

Usage:
    python benchmarks/benchmark_speed.py
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import FastICA as SklearnFastICA

from fastica_torch import FastICA


def benchmark_single(
    X_np: np.ndarray,
    X_torch: torch.Tensor,
    n_components: int,
    n_warmup: int = 1,
    n_runs: int = 3
) -> Dict[str, float]:
    """
    Benchmark a single configuration.
    
    Args:
        X_np: NumPy array for sklearn
        X_torch: PyTorch tensor for fastica_torch
        n_components: Number of ICA components
        n_warmup: Number of warmup runs (not timed)
        n_runs: Number of timed runs
        
    Returns:
        Dict with timing results
    """
    results = {}
    
    # -----------------------------
    # sklearn benchmark
    # -----------------------------
    sklearn_times = []
    for i in range(n_warmup + n_runs):
        sklearn_ica = SklearnFastICA(
            n_components=n_components,
            random_state=42,
            max_iter=200
        )
        start = time.perf_counter()
        sklearn_ica.fit_transform(X_np)
        elapsed = time.perf_counter() - start
        
        if i >= n_warmup:
            sklearn_times.append(elapsed)
    
    results["sklearn_mean"] = np.mean(sklearn_times)
    results["sklearn_std"] = np.std(sklearn_times)
    
    # -----------------------------
    # fastica_torch CPU benchmark
    # -----------------------------
    torch_times = []
    for i in range(n_warmup + n_runs):
        torch_ica = FastICA(
            n_components=n_components,
            random_state=42,
            max_iter=200
        )
        start = time.perf_counter()
        torch_ica.fit_transform(X_torch)
        elapsed = time.perf_counter() - start
        
        if i >= n_warmup:
            torch_times.append(elapsed)
    
    results["torch_cpu_mean"] = np.mean(torch_times)
    results["torch_cpu_std"] = np.std(torch_times)
    
    # Speedup
    results["speedup"] = results["sklearn_mean"] / results["torch_cpu_mean"]
    
    return results


def run_benchmarks() -> List[Dict]:
    """Run benchmarks across various configurations."""
    
    configurations = [
        # (n_samples, n_features, n_components, description)
        (500, 50, 10, "Small"),
        (1000, 100, 20, "Medium-small"),
        (5000, 100, 20, "Medium"),
        (5000, 200, 50, "Medium-large"),
        (20000, 100, 20, "Large samples"),
        (1000, 500, 50, "Wide features"),
        (500, 1000, 50, "Very wide (n < d)"),
    ]
    
    all_results = []
    
    print("=" * 80)
    print("FastICA Speed Benchmark: fastica_torch vs sklearn")
    print("=" * 80)
    print(f"{'Config':<20} {'Shape':<20} {'sklearn (s)':<15} {'torch CPU (s)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for n_samples, n_features, n_components, desc in configurations:
        # Generate data
        np.random.seed(42)
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        X_torch = torch.from_numpy(X_np)
        
        # Run benchmark
        results = benchmark_single(X_np, X_torch, n_components)
        
        # Store results
        results["description"] = desc
        results["n_samples"] = n_samples
        results["n_features"] = n_features
        results["n_components"] = n_components
        all_results.append(results)
        
        # Print results
        shape_str = f"({n_samples}, {n_features}) -> {n_components}"
        sklearn_str = f"{results['sklearn_mean']:.4f} ± {results['sklearn_std']:.4f}"
        torch_str = f"{results['torch_cpu_mean']:.4f} ± {results['torch_cpu_std']:.4f}"
        speedup_str = f"{results['speedup']:.2f}x"
        
        print(f"{desc:<20} {shape_str:<20} {sklearn_str:<15} {torch_str:<15} {speedup_str:<10}")
    
    print("=" * 80)
    
    return all_results


def generate_markdown_table(results: List[Dict]) -> str:
    """Generate a markdown table from benchmark results."""
    
    lines = [
        "# FastICA Speed Benchmark Results",
        "",
        "Comparison of `fastica_torch` (PyTorch CPU) vs `sklearn.decomposition.FastICA`.",
        "",
        "| Configuration | Shape | sklearn (s) | torch CPU (s) | Speedup |",
        "|--------------|-------|-------------|---------------|---------|",
    ]
    
    for r in results:
        shape = f"({r['n_samples']}, {r['n_features']}) → {r['n_components']}"
        sklearn = f"{r['sklearn_mean']:.4f}"
        torch_cpu = f"{r['torch_cpu_mean']:.4f}"
        speedup = f"{r['speedup']:.2f}x"
        lines.append(f"| {r['description']} | {shape} | {sklearn} | {torch_cpu} | {speedup} |")
    
    lines.extend([
        "",
        "## Notes",
        "",
        "- Times are mean of 3 runs after 1 warmup run",
        "- Speedup > 1.0 means fastica_torch is faster",
        "- Both using `random_state=42`, `max_iter=200`, `algorithm='parallel'`",
        "- Data type: float32",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_benchmarks()
    
    # Generate markdown report
    markdown = generate_markdown_table(results)
    
    # Save to file
    with open("benchmarks/BENCHMARK_RESULTS.md", "w") as f:
        f.write(markdown)
    
    print("\nBenchmark results saved to benchmarks/BENCHMARK_RESULTS.md")
