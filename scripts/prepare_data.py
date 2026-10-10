"""Reshape the cleaned Forest and Carbon dataset into one row per country, indicator and year.

Input (wide):  one row per country x indicator, with a column per year (F1992 ... F2020).
Output (long): country, area_type, indicator, unit, year, value

Why: one fact per row ("Ireland, Forest area, 2010") is easier to retrieve and cite.
Run from the project root:  python scripts/prepare_data.py
"""

import re
import sys

import pandas as pd

RAW = "data/raw/forest_carbon_cleaned.csv"
OUT = "data/forest_carbon_long.csv"
# Aggregates whose ISO3 code does not end in "TMP"
GROUP_CODES = {"EMU", "G20", "G7", "WLD"}


def prepare(raw_path: str = RAW, out_path: str = OUT) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    year_cols = [c for c in df.columns if re.fullmatch(r"F\d{4}", c)]

    # Regional and economic groups (Africa, G20, World ...) are kept but labelled, so
    # "which country lost the most forest?" compares countries, not continents.
    is_group = df["ISO3"].str.endswith("TMP") | df["ISO3"].isin(GROUP_CODES)
    df["area_type"] = is_group.map({True: "regional group", False: "country"})

    long = df.melt(
        id_vars=["Country", "area_type", "Indicator", "Unit"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    long["year"] = long["year"].str[1:].astype(int)
    long["value"] = long["value"].round(2)
    long = long.rename(columns={"Country": "country", "Indicator": "indicator", "Unit": "unit"})
    long = long.dropna(subset=["value"]).sort_values(["country", "indicator", "year"])
    long.to_csv(out_path, index=False)
    return long


if __name__ == "__main__":
    out = prepare(*sys.argv[1:])
    print(f"Wrote {len(out):,} rows to {OUT}")
