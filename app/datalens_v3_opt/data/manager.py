"""
Dataset management: DatasetEntry (a single loaded dataset) and DatasetManager
(the collection that handles multi-dataset operations and merging).
"""

from __future__ import annotations

import uuid

import pandas as pd

from datalens_v3_opt.data.core_columns import CORE_COLUMNS


class DatasetEntry:
    """One loaded dataset together with its canonical column mapping."""

    def __init__(self, name: str, df: pd.DataFrame, col_map: dict,
                 source_file: str = ""):
        self.uid         = str(uuid.uuid4())
        self.name        = name
        self.df          = df            # full DataFrame (original or renamed cols)
        self.col_map     = col_map       # {canonical_field: original_col_name}
        self.source_file = source_file
        self.row_count   = len(df)
        self.col_count   = len(df.columns)

    @property
    def mapped_fields(self) -> set:
        return set(self.col_map.keys())


class DatasetManager:
    """
    Manages a collection of DatasetEntry objects and handles merging/concatenation.
    """

    def __init__(self):
        self.entries: list[DatasetEntry] = []

    def add(self, entry: DatasetEntry):
        """Add an entry, deduplicating the name if needed."""
        names = {e.name for e in self.entries}
        if entry.name in names:
            entry.name += f" ({entry.uid})"
        self.entries.append(entry)

    def remove(self, uid: str):
        self.entries = [e for e in self.entries if e.uid != uid]

    def clear(self):
        self.entries.clear()

    def suggest_merge(self) -> dict:
        """
        Return alignment metadata for the merge wizard::

            {
              "field_coverage": {canonical: {entry_name: orig_col, ...}, ...},
              "unmatched":      {entry_name: [col, ...], ...},
            }
        """
        coverage: dict = {}
        for field in CORE_COLUMNS:
            row: dict = {}
            for e in self.entries:
                if field in e.col_map:
                    row[e.name] = e.col_map[field]
            if row:
                coverage[field] = row

        unmatched: dict = {}
        for e in self.entries:
            mapped_orig = set(e.col_map.values())
            rest = [c for c in e.df.columns if c not in mapped_orig]
            if rest:
                unmatched[e.name] = rest

        return {"field_coverage": coverage, "unmatched": unmatched}

    def merge(self, fields: list) -> pd.DataFrame:
        """
        Produce a combined DataFrame with one column per canonical field.
        Missing columns are filled with pd.NA.  A ``_source_dataset`` column
        is always appended.
        """
        parts = []
        for e in self.entries:
            n = e.row_count
            chunk: dict = {}
            for f in fields:
                if f in e.df.columns:
                    chunk[f] = e.df[f].values
                elif f in e.col_map:
                    orig = e.col_map[f]
                    chunk[f] = e.df[orig].values if orig in e.df.columns else [pd.NA] * n
                else:
                    chunk[f] = [pd.NA] * n
            chunk["_source_dataset"] = [e.name] * n
            parts.append(pd.DataFrame(chunk))
        return pd.concat(parts, ignore_index=True)
