import polars as pl

def fft_direct(series: pl.Series) -> tuple[pl.Series, pl.Series]:
    """Direct forward FFT bypassing polars plugin dispatch.

    Takes a polars Series of f64, returns a tuple of (re, im) Series.
    """
    ...

def ifft_direct(re_series: pl.Series, im_series: pl.Series) -> tuple[pl.Series, pl.Series]:
    """Direct inverse FFT bypassing polars plugin dispatch.

    Takes two polars Series (re, im) of f64, returns a tuple of (re, im) Series.
    Output is normalized by 1/N.
    """
    ...
