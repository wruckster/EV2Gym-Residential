#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from typing import Iterable, List, Optional

import pandas as pd


def iter_csv_files(inputs: List[str], glob_pattern: Optional[str]) -> List[str]:
    files: List[str] = []
    for inp in inputs:
        if os.path.isdir(inp):
            if glob_pattern is None:
                # default: all csv files
                pat = re.compile(r".*\.csv$", re.IGNORECASE)
                for name in os.listdir(inp):
                    path = os.path.join(inp, name)
                    if os.path.isfile(path) and pat.match(name):
                        files.append(path)
            else:
                from glob import glob

                files.extend(glob(os.path.join(inp, glob_pattern)))
        elif os.path.isfile(inp):
            files.append(inp)
        else:
            raise FileNotFoundError(f"Input not found: {inp}")
    # de-dup and preserve order
    seen = set()
    out: List[str] = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def extract_household_id(path: str, regex: Optional[str], seq_id: Optional[int]) -> int:
    if regex:
        m = re.search(regex, os.path.basename(path))
        if not m:
            raise ValueError(f"Filename does not match regex for id extraction: {path}")
        gid = m.group(1) if m.groups() else m.group(0)
        try:
            return int(gid)
        except Exception:
            # hash-like stable mapping for non-integer ids
            return abs(hash(gid)) % 1_000_000_000
    assert seq_id is not None
    return seq_id


def read_household_csv(
    path: str,
    timestamp_col: str,
    demand_col: str,
    solar_col: Optional[str],
    tz: Optional[str],
) -> pd.DataFrame:
    usecols = [timestamp_col, demand_col]
    attempted_with_solar = False
    if solar_col:
        usecols_with_solar = [timestamp_col, demand_col, solar_col]
        attempted_with_solar = True
        try:
            df = pd.read_csv(path, usecols=usecols_with_solar, parse_dates=[timestamp_col])
        except ValueError:
            # Solar column missing in this CSV; fall back to reading without it
            df = pd.read_csv(path, usecols=usecols, parse_dates=[timestamp_col])
            solar_col = None  # signal missing
    else:
        df = pd.read_csv(path, usecols=usecols, parse_dates=[timestamp_col])
    df = df.rename(columns={timestamp_col: "timestamp", demand_col: "demand"})
    if solar_col and solar_col in df.columns:
        df = df.rename(columns={solar_col: "solar"})
    else:
        df["solar"] = 0.0
    # Ensure numeric
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)
    df["solar"] = pd.to_numeric(df["solar"], errors="coerce").fillna(0.0)
    # Timezone handling (handle DST ambiguities robustly)
    if tz:
        # Localize naive timestamps; resolve DST issues deterministically
        if df["timestamp"].dt.tz is None:
            try:
                df["timestamp"] = df["timestamp"].dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                # Fallback: mark ambiguous/nonexistent as NaT then forward-fill
                df["timestamp"] = df["timestamp"].dt.tz_localize(tz, ambiguous=False, nonexistent="NaT").fillna(method="ffill")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    return df[["timestamp", "demand", "solar"]]


def rollup(
    inputs: List[str],
    output_parquet: str,
    id_regex: Optional[str],
    timestamp_col: str,
    demand_col: str,
    solar_col: Optional[str],
    tz: Optional[str],
) -> None:
    files = iter_csv_files(inputs, glob_pattern=None)
    if not files:
        raise RuntimeError("No input CSV files found")

    records: List[pd.DataFrame] = []
    seq_id = 0
    for f in files:
        hid = extract_household_id(f, id_regex, seq_id)
        if id_regex is None:
            seq_id += 1
        df = read_household_csv(f, timestamp_col, demand_col, solar_col, tz)
        df.insert(1, "household_id", hid)
        records.append(df)

    big = pd.concat(records, axis=0, ignore_index=True)
    # Sort and ensure dtypes
    big = big.sort_values(["timestamp", "household_id"]).reset_index(drop=True)
    big["household_id"] = big["household_id"].astype("int32")

    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        engine = "pyarrow"
    except Exception:
        engine = "auto"
    big.to_parquet(output_parquet, engine=engine, index=False)
    print(f"Wrote {len(big):,} rows across {len(files)} households to {output_parquet}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Roll up household CSVs into a single Parquet with household_id.")
    ap.add_argument("inputs", nargs="+", help="CSV files and/or directories containing CSVs")
    ap.add_argument("--out", required=True, help="Output Parquet path")
    ap.add_argument("--id-from-filename-regex", default=r"(\d+)", help="Regex with a capturing group for household id (default: first integer in filename)")
    ap.add_argument("--timestamp-col", default="interval_start", help="Timestamp column name in CSV (default: interval_start)")
    ap.add_argument("--demand-col", default="demand", help="Demand column name in CSV (default: demand)")
    ap.add_argument("--solar-col", default="solar", help="Solar column name in CSV (default: solar). If missing in a file, defaults to 0")
    ap.add_argument("--tz", default=None, help="Timezone to localize/convert timestamps (e.g., Australia/Sydney)")

    args = ap.parse_args()
    rollup(
        inputs=args.inputs,
        output_parquet=args.out,
        id_regex=args.id_from_filename_regex,
        timestamp_col=args.timestamp_col,
        demand_col=args.demand_col,
        solar_col=args.solar_col,
        tz=args.tz,
    )


if __name__ == "__main__":
    main()
