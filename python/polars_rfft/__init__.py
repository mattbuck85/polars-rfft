from __future__ import annotations

from polars_rfft._polars_rfft import fft_direct, ifft_direct
from polars_rfft.fft import RfftNamespace

rfft = RfftNamespace

__all__ = ["RfftNamespace", "rfft", "fft_direct", "ifft_direct"]
