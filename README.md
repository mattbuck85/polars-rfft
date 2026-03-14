# polars-rfft

FFT as native Polars expressions. Built with Rust ([RustFFT](https://github.com/ejmahler/RustFFT)) and PyO3 for zero-copy, vectorized computation over Polars DataFrames.

## Install

```bash
pip install polars-rfft
```

## Quick start

```python
import polars as pl
from polars_rfft import rfft

df = pl.DataFrame({"signal": [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]})

# Forward FFT — returns struct {re, im}
df.with_columns(spectrum=rfft(pl.col("signal")).fft())
```

### Roundtrip: FFT → IFFT

```python
df.with_columns(
    recovered=rfft(rfft(pl.col("signal")).fft()).ifft_real()
)
```

### Magnitude, phase, power spectrum

```python
fft_expr = rfft(pl.col("signal")).fft()
df.with_columns(
    mag=rfft(fft_expr).magnitude(),
    phase=rfft(fft_expr).phase(),
    power=rfft(fft_expr).power_spectrum(),
)
```

### Complex-valued input

```python
df = pl.DataFrame({
    "z": [{"re": 1.0, "im": 0.0}, {"re": 0.0, "im": 1.0},
           {"re": -1.0, "im": 0.0}, {"re": 0.0, "im": -1.0}]
})
df.with_columns(spectrum=rfft(pl.col("z")).fft_complex())
```

## API reference

Complex values are represented as struct columns with fields `re` (real) and `im` (imaginary).

| Method | Input | Output | Description |
|-|-|-|-|
| `fft()` | f64 | struct{re,im} | Forward FFT on real signal |
| `fft_complex()` | struct{re,im} | struct{re,im} | Forward FFT on complex signal |
| `ifft()` | struct{re,im} | struct{re,im} | Inverse FFT (normalized by 1/N) |
| `ifft_real()` | struct{re,im} | f64 | Inverse FFT, real part only |
| `magnitude()` | struct{re,im} | f64 | |z| = sqrt(re² + im²) |
| `phase()` | struct{re,im} | f64 | atan2(im, re) in radians |
| `power_spectrum()` | struct{re,im} | f64 | |z|² = re² + im² |

All methods are accessed via the `rfft()` wrapper:

```python
from polars_rfft import rfft

rfft(pl.col("signal")).fft()          # forward
rfft(pl.col("spectrum")).ifft()       # inverse
rfft(pl.col("spectrum")).magnitude()  # magnitude
```

### Direct functions

For lower overhead on smaller signals, `fft_direct` and `ifft_direct` bypass the Polars expression engine and operate directly on Series:

```python
from polars_rfft import fft_direct, ifft_direct

re, im = fft_direct(df["signal"])           # Series → (Series, Series)
re, im = ifft_direct(re, im)               # (Series, Series) → (Series, Series)
```

## Performance

Benchmarked on x86 arch end-to-end against numpy on signals of varying length. Median of 7 runs after warmup.

### Expression API (`rfft().fft()`)

| Signal length | Operation | polars-rfft | numpy | Speedup |
|-|-|-|-|-|
| 1,024 | fft | 0.14 ms | 0.02 ms | 0.2x |
| 4,096 | fft | 0.24 ms | 0.08 ms | 0.3x |
| 16,384 | fft | 0.66 ms | 0.99 ms | **1.5x** |
| 65,536 | fft | 2.2 ms | 4.7 ms | **2.1x** |
| 262,144 | fft | 10.2 ms | 22.1 ms | **2.2x** |
| 1,048,576 | fft | 33.9 ms | 53.3 ms | **1.9x** |

### Direct API (`fft_direct()`)

Bypasses Polars expression dispatch for ~0.1ms less overhead:

| Signal length | Operation | fft_direct | numpy | Speedup |
|-|-|-|-|-|
| 64 | fft | 0.014 ms | 0.013 ms | 0.9x |
| 256 | fft | 0.018 ms | 0.013 ms | 0.7x |
| 1,024 | fft | 0.040 ms | 0.022 ms | 0.6x |
| 4,096 | fft | 0.14 ms | 0.08 ms | 0.5x |
| 16,384 | fft | 0.64 ms | 0.99 ms | **1.5x** |
| 65,536 | fft | 2.1 ms | 4.7 ms | **2.3x** |
| 262,144 | fft | 9.9 ms | 22.1 ms | **2.2x** |
| 1,048,576 | fft | 27.9 ms | 53.3 ms | **1.9x** |

At small sizes (< 4K), numpy is faster due to its highly optimized C/Fortran backend. At 16K+ elements, RustFFT's radix algorithms dominate — up to 2.3x faster than numpy. The direct API eliminates ~0.1ms of Polars expression dispatch overhead, matching numpy at very small sizes.

Run it yourself:

```bash
pip install numpy
python benchmarks/bench_fft.py --sizes 1024 16384 262144 1048576
```

## License

MIT
