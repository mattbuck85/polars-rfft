from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from polars_rfft import rfft


def _unpack_struct(series: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a struct {re, im} series into numpy arrays."""
    df = series.struct.unnest()
    return df["re"].to_numpy(), df["im"].to_numpy()


def _to_complex(series: pl.Series) -> np.ndarray:
    """Convert struct {re, im} series to numpy complex array."""
    re, im = _unpack_struct(series)
    return re + 1j * im


# ── Forward FFT parity with numpy ──


class TestFFTForward:
    @pytest.mark.parametrize("n", [4, 8, 16, 64, 256, 1024])
    def test_fft_real_vs_numpy(self, n: int):
        """Forward FFT of real signal matches numpy.fft.fft."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(n)
        expected = np.fft.fft(signal)

        df = pl.DataFrame({"x": signal.tolist()})
        result = _to_complex(df.select(rfft(pl.col("x")).fft())["x"])

        assert_allclose(result.real, expected.real, atol=1e-10)
        assert_allclose(result.imag, expected.imag, atol=1e-10)

    def test_dc_signal(self):
        signal = np.array([3.0] * 8)
        expected = np.fft.fft(signal)

        df = pl.DataFrame({"x": signal.tolist()})
        result = _to_complex(df.select(rfft(pl.col("x")).fft())["x"])

        assert_allclose(result.real, expected.real, atol=1e-10)
        assert_allclose(result.imag, expected.imag, atol=1e-10)

    def test_impulse(self):
        signal = np.zeros(8)
        signal[0] = 1.0
        expected = np.fft.fft(signal)

        df = pl.DataFrame({"x": signal.tolist()})
        result = _to_complex(df.select(rfft(pl.col("x")).fft())["x"])

        assert_allclose(result.real, expected.real, atol=1e-10)
        assert_allclose(result.imag, expected.imag, atol=1e-10)

    def test_sinusoid(self):
        n = 16
        k = 3
        signal = np.sin(2 * np.pi * k * np.arange(n) / n)
        expected = np.fft.fft(signal)

        df = pl.DataFrame({"x": signal.tolist()})
        result = _to_complex(df.select(rfft(pl.col("x")).fft())["x"])

        assert_allclose(result.real, expected.real, atol=1e-10)
        assert_allclose(result.imag, expected.imag, atol=1e-10)

    def test_non_power_of_two(self):
        """FFT works for non-power-of-2 lengths."""
        rng = np.random.default_rng(123)
        for n in [3, 7, 13, 100, 997]:
            signal = rng.standard_normal(n)
            expected = np.fft.fft(signal)

            df = pl.DataFrame({"x": signal.tolist()})
            result = _to_complex(df.select(rfft(pl.col("x")).fft())["x"])

            assert_allclose(result.real, expected.real, atol=1e-8, err_msg=f"n={n}")
            assert_allclose(result.imag, expected.imag, atol=1e-8, err_msg=f"n={n}")

    def test_empty(self):
        df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
        result = df.select(rfft(pl.col("x")).fft())
        assert len(result) == 0


# ── Complex FFT parity with numpy ──


class TestFFTComplex:
    @pytest.mark.parametrize("n", [4, 16, 64, 256])
    def test_fft_complex_vs_numpy(self, n: int):
        """Forward FFT of complex signal matches numpy.fft.fft."""
        rng = np.random.default_rng(99)
        signal = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        expected = np.fft.fft(signal)

        structs = [{"re": float(z.real), "im": float(z.imag)} for z in signal]
        df = pl.DataFrame({"z": structs})
        result = _to_complex(df.select(rfft(pl.col("z")).fft_complex())["z"])

        assert_allclose(result.real, expected.real, atol=1e-10)
        assert_allclose(result.imag, expected.imag, atol=1e-10)


# ── Inverse FFT parity with numpy ──


class TestIFFT:
    @pytest.mark.parametrize("n", [4, 8, 16, 64, 256, 1024])
    def test_ifft_vs_numpy(self, n: int):
        """Inverse FFT matches numpy.fft.ifft."""
        rng = np.random.default_rng(77)
        signal = rng.standard_normal(n)
        spectrum = np.fft.fft(signal)
        expected = np.fft.ifft(spectrum)

        structs = [{"re": float(z.real), "im": float(z.imag)} for z in spectrum]
        df = pl.DataFrame({"f": structs})
        result = _to_complex(df.select(rfft(pl.col("f")).ifft())["f"])

        assert_allclose(result.real, expected.real, atol=1e-10)
        assert_allclose(result.imag, expected.imag, atol=1e-10)

    @pytest.mark.parametrize("n", [4, 8, 16, 64, 256, 1024])
    def test_roundtrip_real(self, n: int):
        """fft → ifft_real recovers original signal."""
        rng = np.random.default_rng(55)
        signal = rng.standard_normal(n)

        df = pl.DataFrame({"x": signal.tolist()})
        recovered = df.select(
            rfft(rfft(pl.col("x")).fft()).ifft_real().alias("r")
        )["r"].to_numpy()

        assert_allclose(recovered, signal, atol=1e-10)

    def test_roundtrip_complex(self):
        """fft → ifft recovers original (imaginary part ~0 for real input)."""
        signal = np.array([1.0, -1.0, 2.0, -2.0])
        df = pl.DataFrame({"x": signal.tolist()})
        result = _to_complex(
            df.select(rfft(rfft(pl.col("x")).fft()).ifft().alias("r"))["r"]
        )

        assert_allclose(result.real, signal, atol=1e-10)
        assert_allclose(result.imag, np.zeros_like(signal), atol=1e-10)


# ── Magnitude parity with numpy ──


class TestMagnitude:
    @pytest.mark.parametrize("n", [4, 16, 64, 256])
    def test_magnitude_vs_numpy(self, n: int):
        """Magnitude matches np.abs(np.fft.fft(signal))."""
        rng = np.random.default_rng(33)
        signal = rng.standard_normal(n)
        expected = np.abs(np.fft.fft(signal))

        df = pl.DataFrame({"x": signal.tolist()})
        result = df.select(
            rfft(rfft(pl.col("x")).fft()).magnitude().alias("mag")
        )["mag"].to_numpy()

        assert_allclose(result, expected, atol=1e-10)

    def test_magnitude_manual(self):
        df = pl.DataFrame({"z": [{"re": 3.0, "im": 4.0}, {"re": 0.0, "im": 1.0}]})
        result = df.select(rfft(pl.col("z")).magnitude().alias("mag"))["mag"].to_numpy()
        assert_allclose(result, [5.0, 1.0])


# ── Phase parity with numpy ──


class TestPhase:
    @pytest.mark.parametrize("n", [4, 16, 64, 256])
    def test_phase_vs_numpy(self, n: int):
        """Phase matches np.angle(np.fft.fft(signal))."""
        rng = np.random.default_rng(11)
        signal = rng.standard_normal(n)
        expected = np.angle(np.fft.fft(signal))

        df = pl.DataFrame({"x": signal.tolist()})
        result = df.select(
            rfft(rfft(pl.col("x")).fft()).phase().alias("ph")
        )["ph"].to_numpy()

        assert_allclose(result, expected, atol=1e-10)

    def test_phase_known(self):
        df = pl.DataFrame(
            {"z": [{"re": 1.0, "im": 0.0}, {"re": 0.0, "im": 1.0}, {"re": -1.0, "im": 0.0}]}
        )
        result = df.select(rfft(pl.col("z")).phase().alias("ph"))["ph"].to_numpy()
        assert_allclose(result, [0.0, math.pi / 2, math.pi])


# ── Power spectrum parity with numpy ──


class TestPowerSpectrum:
    @pytest.mark.parametrize("n", [4, 16, 64, 256])
    def test_power_vs_numpy(self, n: int):
        """Power spectrum matches np.abs(np.fft.fft(signal))**2."""
        rng = np.random.default_rng(22)
        signal = rng.standard_normal(n)
        expected = np.abs(np.fft.fft(signal)) ** 2

        df = pl.DataFrame({"x": signal.tolist()})
        result = df.select(
            rfft(rfft(pl.col("x")).fft()).power_spectrum().alias("pow")
        )["pow"].to_numpy()

        assert_allclose(result, expected, atol=1e-8)

    def test_power_impulse(self):
        """Impulse → flat power spectrum, each bin = 1.0."""
        signal = np.zeros(4)
        signal[0] = 1.0
        expected = np.abs(np.fft.fft(signal)) ** 2

        df = pl.DataFrame({"x": signal.tolist()})
        result = df.select(
            rfft(rfft(pl.col("x")).fft()).power_spectrum().alias("pow")
        )["pow"].to_numpy()

        assert_allclose(result, expected, atol=1e-10)
