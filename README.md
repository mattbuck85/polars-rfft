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

## Performance

Benchmarked end-to-end against numpy on signals of varying length. Median of 5 runs after warmup.

| Signal length | Operation | polars-rfft | numpy | Speedup |
|-|-|-|-|-|
| 1,024 | fft | 0.2 ms | 0.0 ms | 0.1x |
| 1,024 | ifft | 0.1 ms | 0.0 ms | 0.1x |
| 4,096 | fft | 0.2 ms | 0.0 ms | 0.2x |
| 4,096 | ifft | 0.2 ms | 0.0 ms | 0.2x |
| 16,384 | fft | 0.5 ms | 0.5 ms | 1.0x |
| 16,384 | ifft | 0.4 ms | 0.2 ms | 0.4x |
| 65,536 | fft | 1.2 ms | 2.6 ms | **2.1x** |
| 65,536 | ifft | 1.2 ms | 1.5 ms | **1.2x** |
| 262,144 | fft | 6.3 ms | 12.3 ms | **2.0x** |
| 262,144 | ifft | 6.7 ms | 7.8 ms | **1.2x** |
| 1,048,576 | fft | 30.9 ms | 57.3 ms | **1.9x** |
| 1,048,576 | ifft | 31.8 ms | 41.4 ms | **1.3x** |

At small sizes (< 16K), numpy is faster due to Polars expression dispatch overhead. At 64K+ elements, RustFFT's optimized radix algorithms dominate — forward FFT is ~2x faster than numpy.

Run it yourself:

```bash
pip install numpy
python benchmarks/bench_fft.py --sizes 1024 16384 262144 1048576
```

## License

MIT
