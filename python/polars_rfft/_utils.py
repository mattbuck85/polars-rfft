from __future__ import annotations

import polars as pl
from pathlib import Path
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent


def pl_plugin(*, symbol: str, args: list[pl.Expr], is_elementwise: bool = False) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        function_name=symbol,
        args=args,
        kwargs={},
        is_elementwise=is_elementwise,
    )
