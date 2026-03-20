"""
File loading and parsing utilities.

Covers encoding detection, delimiter sniffing, table-start detection,
date parsing, and hashtag extraction.
"""

import re
import csv

import pandas as pd

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False


def detect_encoding(path: str) -> str:
    """Return the most likely character encoding for *path*."""
    if HAS_CHARDET:
        with open(path, "rb") as f:
            raw = f.read(65536)
        result = chardet.detect(raw)
        return result.get("encoding") or "utf-8"
    return "utf-8"


def read_raw_lines(path: str, enc: str, n: int = 300) -> list:
    """Read up to *n* lines from *path* using the given encoding."""
    lines = []
    try:
        with open(path, encoding=enc, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line.rstrip("\n"))
    except Exception:
        pass
    return lines


def detect_delimiter(lines: list, skip: int = 0) -> str:
    """Guess the column delimiter by counting candidate characters in data lines."""
    candidates = [",", ";", "\t", "|"]
    data_lines = lines[skip:skip + 10]
    counts = {d: sum(line.count(d) for line in data_lines) for d in candidates}
    return max(counts, key=counts.get)


def field_count(line: str, delim: str) -> int:
    """Return the number of fields in *line* using *delim*."""
    try:
        return len(next(csv.reader([line], delimiter=delim)))
    except Exception:
        return 0


def detect_table_start(lines: list, delim: str) -> int:
    """Return the 0-based index of the header line."""
    if not lines:
        return 0
    target = max(field_count(l, delim) for l in lines[:30])
    for i, line in enumerate(lines[:30]):
        if field_count(line, delim) >= target * 0.8:
            return i
    return 0


def extract_hashtags(text: str) -> list:
    """Extract all hashtag words from *text* (without the # symbol)."""
    return re.findall(r"#(\w+)", str(text))


def safe_date_parse(series: pd.Series) -> pd.Series:
    """Parse a Series of date strings to UTC-aware Timestamps, coercing errors."""
    try:
        return pd.to_datetime(series, utc=True, errors="coerce")
    except Exception:
        return series
