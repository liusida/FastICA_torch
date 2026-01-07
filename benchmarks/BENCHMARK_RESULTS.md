# FastICA Speed Benchmark Results

Comparison of `fastica_torch` (PyTorch CPU) vs `sklearn.decomposition.FastICA`.

| Configuration | Shape | sklearn (s) | torch CPU (s) | Speedup |
|--------------|-------|-------------|---------------|---------|
| Small | (500, 50) → 10 | 0.0205 | 0.0206 | 1.00x |
| Medium-small | (1000, 100) → 20 | 0.1059 | 0.0747 | 1.42x |
| Medium | (5000, 100) → 20 | 0.2551 | 0.0804 | 3.18x |
| Medium-large | (5000, 200) → 50 | 0.5993 | 0.2126 | 2.82x |
| Large samples | (20000, 100) → 20 | 1.0608 | 0.2576 | 4.12x |
| Wide features | (1000, 500) → 50 | 0.2855 | 0.1165 | 2.45x |
| Very wide (n < d) | (500, 1000) → 50 | 0.2664 | 0.1055 | 2.53x |

## Notes

- Times are mean of 3 runs after 1 warmup run
- Speedup > 1.0 means fastica_torch is faster
- Both using `random_state=42`, `max_iter=200`, `algorithm='parallel'`
- Data type: float32