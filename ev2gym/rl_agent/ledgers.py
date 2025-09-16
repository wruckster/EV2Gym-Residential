from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterable
import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    pd = None  # type: ignore

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    pl = None  # type: ignore


@dataclass
class ColumnSpec:
    name: str
    dtype: np.dtype


class _BaseLedgerBuffers:
    """
    NumPy-backed columnar buffers for time-indexed ledgers.
    - Preallocate arrays for each column with shape (T,)
    - Provide row-wise set API and exporters to Pandas/Polars
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        columns: Iterable[ColumnSpec],
    ) -> None:
        assert timestamps.ndim == 1, "timestamps must be 1D"
        assert np.issubdtype(timestamps.dtype, np.datetime64), "timestamps must be datetime64"
        self.T: int = int(timestamps.shape[0])
        self.timestamps: np.ndarray = timestamps.astype("datetime64[ns]")

        # Fixed column order
        self._col_specs: List[ColumnSpec] = [ColumnSpec("timestamp", np.dtype("datetime64[ns]"))]
        self._col_specs.extend(columns)

        # Storage
        self._data: Dict[str, np.ndarray] = {
            "timestamp": self.timestamps.copy(),
        }
        for spec in columns:
            # Choose a safe default fill value per dtype to avoid casting warnings
            dt = spec.dtype
            if np.issubdtype(dt, np.floating):
                fill = np.nan
            elif np.issubdtype(dt, np.integer):
                # Use 0 as a neutral default for integer columns
                fill = 0
            elif np.issubdtype(dt, np.bool_):
                fill = False
            else:
                # Fallback: try NaN, most non-int numeric types will accept it
                fill = np.nan
            self._data[spec.name] = np.full(self.T, fill, dtype=dt)

        # Name -> index mapping for faster vectorization
        self._name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.columns)}

    @property
    def columns(self) -> List[str]:
        return [spec.name for spec in self._col_specs]

    def set_row(self, t: int, values: Dict[str, Any]) -> None:
        """Set multiple columns at row t. Missing keys are ignored."""
        if t < 0 or t >= self.T:
            raise IndexError(f"row index out of bounds: {t}")
        for k, v in values.items():
            if k == "timestamp":  # immutable per-row
                continue
            arr = self._data.get(k)
            if arr is None:
                # Silently ignore unknown columns to allow incremental rollout
                continue
            arr[t] = v

    def to_pandas(self):  # type: ignore
        if pd is None:
            raise RuntimeError("pandas is not available")
        data = {name: self._data[name] for name in self.columns if name in self._data}
        return pd.DataFrame(data)

    def to_polars(self):  # type: ignore
        if pl is None:
            raise RuntimeError("polars is not available")
        # Polars accepts numpy arrays directly
        return pl.DataFrame({name: self._data[name] for name in self.columns if name in self._data})

    def to_parquet(self, path: str) -> None:
        if pl is not None:
            self.to_polars().write_parquet(path)  # type: ignore
            return
        if pd is not None:
            # pandas -> pyarrow
            self.to_pandas().to_parquet(path)  # type: ignore
            return
        raise RuntimeError("Neither polars nor pandas is available to write parquet")


class GlobalLedgerBuffers(_BaseLedgerBuffers):
    """Buffers for global, shared signals (prices, weather, DR flags, etc.)."""
    pass


class AccountLedgerBuffers(_BaseLedgerBuffers):
    """Buffers for per-account signals (household demand, PV, EV/charger state)."""
    def __init__(
        self,
        account_id: int,
        timestamps: np.ndarray,
        columns: Iterable[ColumnSpec],
    ) -> None:
        self.account_id: int = account_id
        super().__init__(timestamps, columns)
