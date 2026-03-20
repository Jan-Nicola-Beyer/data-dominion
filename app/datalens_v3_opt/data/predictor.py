"""
Column predictor: maps raw DataFrame column names to canonical social-media
fields using sentence-transformers (when available) or fuzzy string matching.
"""

from __future__ import annotations

import re
import difflib

import pandas as pd

from datalens_v3_opt.data.core_columns import CORE_COLUMNS

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


class ColumnPredictor:
    """
    Predicts which canonical social-media field each DataFrame column maps to.
    Uses sentence-transformers when available; otherwise falls back to fuzzy
    string matching plus sample-value heuristics.
    """

    THRESHOLD = 0.42          # minimum confidence to return a match
    SBERT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None
        self._core_embeddings = None
        self._core_names = list(CORE_COLUMNS.keys())
        if HAS_SBERT:
            self._load_sbert()

    def _load_sbert(self):
        try:
            self._model = SentenceTransformer(self.SBERT_MODEL)
            descriptions = [
                f"{name} {info['description']} {' '.join(info['aliases'][:8])}"
                for name, info in CORE_COLUMNS.items()
            ]
            self._core_embeddings = self._model.encode(
                descriptions, normalize_embeddings=True
            )
        except Exception:
            self._model = None

    # ── public API ─────────────────────────────────────────────────────────────
    def predict_all(self, columns: list, sample_values: dict = None) -> dict:
        """Return {col_name: {"field": str|None, "confidence": float, "category": str}}."""
        sv = sample_values or {}
        results = {}
        for col in columns:
            field, conf = self._predict_one(col, sv.get(col, []))
            cat = CORE_COLUMNS[field]["category"] if field else "unknown"
            results[col] = {"field": field, "confidence": conf, "category": cat}
        return results

    def _predict_one(self, col_name: str, samples: list):
        if self._model is not None:
            return self._predict_sbert(col_name, samples)
        return self._predict_fuzzy(col_name, samples)

    # ── sentence-transformers path ─────────────────────────────────────────────
    def _predict_sbert(self, col_name: str, samples: list):
        val_str = ", ".join(str(v)[:40] for v in samples[:3]) if samples else ""
        query = f"column '{col_name}'" + (f" examples: {val_str}" if val_str else "")
        q_emb = self._model.encode([query], normalize_embeddings=True)[0]
        sims = self._core_embeddings @ q_emb          # cosine similarity (unit vectors)
        best_idx = int(np.argmax(sims))
        best_conf = float(sims[best_idx])
        if best_conf >= self.THRESHOLD:
            bonus = self._value_bonus(self._core_names[best_idx], samples)
            final = min(1.0, best_conf * 0.75 + bonus * 0.25)
            return self._core_names[best_idx], final
        return None, best_conf

    # ── fuzzy fallback ─────────────────────────────────────────────────────────
    def _predict_fuzzy(self, col_name: str, samples: list):
        col_norm = re.sub(r"[\s\-\.]+", "_", col_name.lower())
        best_field, best_score = None, 0.0
        for field, info in CORE_COLUMNS.items():
            score = self._alias_score(col_norm, info["aliases"])
            bonus = self._value_bonus(field, samples)
            total = score * 0.75 + bonus * 0.25
            if total > best_score:
                best_score = total
                best_field = field
        if best_score >= self.THRESHOLD:
            return best_field, best_score
        return None, best_score

    @staticmethod
    def _alias_score(col_norm: str, aliases: list) -> float:
        best = 0.0
        for alias in aliases:
            a = re.sub(r"[\s\-\.]+", "_", alias.lower())
            if col_norm == a:
                return 1.0
            if col_norm in a or a in col_norm:
                best = max(best, 0.85)
            ratio = difflib.SequenceMatcher(None, col_norm, a).ratio()
            best = max(best, ratio)
        return best

    @staticmethod
    def _value_bonus(field: str, samples: list) -> float:
        if not samples:
            return 0.0
        hint = CORE_COLUMNS[field]["dtype_hint"]
        score = 0.0
        for v in samples[:5]:
            s = str(v).strip()
            if not s or s.lower() in ("nan", "none", ""):
                continue
            if hint == "url" and s.startswith("http"):
                score = max(score, 1.0)
            elif hint == "datetime":
                try:
                    pd.to_datetime(s)
                    score = max(score, 0.9)
                except Exception:
                    pass
            elif hint == "integer":
                try:
                    int(float(s))
                    score = max(score, 0.7)
                except Exception:
                    pass
            elif hint == "float":
                try:
                    float(s.strip("%"))
                    score = max(score, 0.6)
                except Exception:
                    pass
            elif hint == "boolean":
                if s.lower() in ("true", "false", "yes", "no", "1", "0"):
                    score = max(score, 0.8)
            elif hint == "id":
                if s.isdigit() and len(s) >= 6:
                    score = max(score, 0.5)
        return score


# Global singleton — loaded once per process
_PREDICTOR = None  # type: Optional[ColumnPredictor]


def get_predictor() -> ColumnPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = ColumnPredictor()
    return _PREDICTOR
