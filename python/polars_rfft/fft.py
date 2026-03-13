from __future__ import annotations

import polars as pl

from polars_rfft._utils import pl_plugin


@pl.api.register_expr_namespace("rfft")
class RfftNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    # ── Forward FFT ──

    def fft(self) -> pl.Expr:
        """Forward FFT on a real-valued f64 column.

        Returns a struct column with fields ``re`` and ``im``.
        """
        return pl_plugin(symbol="fft", args=[self._expr])

    def fft_complex(self) -> pl.Expr:
        """Forward FFT on a complex-valued struct {re, im} column.

        Returns a struct column with fields ``re`` and ``im``.
        """
        return pl_plugin(symbol="fft_complex", args=[self._expr])

    # ── Inverse FFT ──

    def ifft(self) -> pl.Expr:
        """Inverse FFT on a struct {re, im} column.

        Returns a struct column with fields ``re`` and ``im`` (normalized by 1/N).
        """
        return pl_plugin(symbol="ifft", args=[self._expr])

    def ifft_real(self) -> pl.Expr:
        """Inverse FFT on a struct {re, im} column, returning only the real part.

        Returns a f64 column (normalized by 1/N). Use when the original signal
        was real-valued.
        """
        return pl_plugin(symbol="ifft_real", args=[self._expr])

    # ── Utilities on complex struct columns ──

    def magnitude(self) -> pl.Expr:
        """Compute magnitude |z| = sqrt(re^2 + im^2) from a struct {re, im} column."""
        return pl_plugin(symbol="magnitude", args=[self._expr], is_elementwise=True)

    def phase(self) -> pl.Expr:
        """Compute phase angle atan2(im, re) in radians from a struct {re, im} column."""
        return pl_plugin(symbol="phase", args=[self._expr], is_elementwise=True)

    def power_spectrum(self) -> pl.Expr:
        """Compute power spectrum |z|^2 = re^2 + im^2 from a struct {re, im} column."""
        return pl_plugin(symbol="power_spectrum", args=[self._expr], is_elementwise=True)
