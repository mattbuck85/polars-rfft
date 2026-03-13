import polars as pl

class RfftNamespace:
    _expr: pl.Expr
    def __init__(self, expr: pl.Expr) -> None: ...

    # Forward FFT
    def fft(self) -> pl.Expr:
        """Forward FFT on a real-valued f64 column.

        Returns a struct column with fields ``re`` and ``im``.
        """
        ...
    def fft_complex(self) -> pl.Expr:
        """Forward FFT on a complex-valued struct {re, im} column.

        Returns a struct column with fields ``re`` and ``im``.
        """
        ...

    # Inverse FFT
    def ifft(self) -> pl.Expr:
        """Inverse FFT on a struct {re, im} column.

        Returns a struct column with fields ``re`` and ``im`` (normalized by 1/N).
        """
        ...
    def ifft_real(self) -> pl.Expr:
        """Inverse FFT on a struct {re, im} column, returning only the real part.

        Returns a f64 column (normalized by 1/N). Use when the original signal
        was real-valued.
        """
        ...

    # Utilities on complex struct columns
    def magnitude(self) -> pl.Expr:
        """Compute magnitude |z| = sqrt(re^2 + im^2) from a struct {re, im} column."""
        ...
    def phase(self) -> pl.Expr:
        """Compute phase angle atan2(im, re) in radians from a struct {re, im} column."""
        ...
    def power_spectrum(self) -> pl.Expr:
        """Compute power spectrum |z|^2 = re^2 + im^2 from a struct {re, im} column."""
        ...
