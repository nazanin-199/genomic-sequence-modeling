"""Engine for per-chromosome GC-stratified negative background sampling."""

import random
from typing import Dict, List
import numpy as np
import pandas as pd
from twobitreader import TwoBitFile


class GCMatchedBackgroundSampler:
    """Samples non-regulatory genomic sequences matching positive GC distributions."""

    def __init__(
        self,
        reference_2bit_path: str,
        buffer_bp: int = 2000,
        window_size: int = 500,
        bin_width: float = 0.02,
        seed: int = 42,
    ):
        self.genome = TwoBitFile(reference_2bit_path)
        self.buffer_bp = buffer_bp
        self.window_size = window_size
        self.bins = np.arange(0.15, 0.90, bin_width)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def build_exclusion_intervals(
        self, df_peaks: pd.DataFrame, canonical_chroms: List[str]
    ) -> Dict[str, np.ndarray]:
        """Merges buffered positive peak coordinates into disjoint exclusion arrays."""
        exclusion = {}
        for chrom in canonical_chroms:
            sub = df_peaks[df_peaks["chrom"] == chrom]
            if len(sub) == 0:
                exclusion[chrom] = np.empty((0, 2), dtype=int)
                continue

            starts = (sub["chromStart"] - self.buffer_bp).clip(lower=0).values
            ends = (sub["chromEnd"] + self.buffer_bp).values

            order = np.argsort(starts)
            starts, ends = starts[order], ends[order]

            merged = []
            cur_s, cur_e = starts[0], ends[0]
            for s, e in zip(starts[1:], ends[1:]):
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    merged.append([cur_s, cur_e])
                    cur_s, cur_e = s, e
            merged.append([cur_s, cur_e])
            exclusion[chrom] = np.array(merged)
        return exclusion

    def is_in_exclusion(self, chrom_intervals: np.ndarray, start: int, end: int) -> bool:
        if len(chrom_intervals) == 0:
            return False
        idx = np.searchsorted(chrom_intervals[:, 0], end, side="right") - 1
        if idx >= 0 and start < chrom_intervals[idx, 1]:
            return True
        idx_next = idx + 1
        if idx_next < len(chrom_intervals) and chrom_intervals[idx_next, 0] < end:
            return True
        return False

    def sample_chromosome_negatives(
        self,
        chrom: str,
        target_gc_series: pd.Series,
        exclusion_intervals: np.ndarray,
        max_attempts_factor: int = 40,
    ) -> List[Dict]:
        """Samples matched null sequences for one chromosome using rejection sampling."""
        chrom_len = len(self.genome[chrom])
        n_needed = len(target_gc_series)

        pos_bin_indices = np.digitize(target_gc_series.values, self.bins)
        needed_per_bin = pd.Series(pos_bin_indices).value_counts().to_dict()
        collected_per_bin = {b: [] for b in needed_per_bin}

        total_collected = 0
        max_attempts = n_needed * max_attempts_factor
        attempts = 0

        while total_collected < n_needed and attempts < max_attempts:
            attempts += 1
            s = random.randint(10000, chrom_len - self.window_size - 10000)
            e = s + self.window_size

            if self.is_in_exclusion(exclusion_intervals, s, e):
                continue

            seq = self.genome[chrom][s:e].upper()
            if seq.count("N") > (self.window_size * 0.05):
                continue

            gc_val = (seq.count("G") + seq.count("C")) / self.window_size
            b_idx = np.digitize([gc_val], self.bins)[0]

            if b_idx in needed_per_bin and len(collected_per_bin[b_idx]) < needed_per_bin[b_idx]:
                collected_per_bin[b_idx].append(
                    {
                        "chrom": chrom,
                        "start": s,
                        "end": e,
                        "center": (s + e) // 2,
                        "seq": seq,
                        "gc": gc_val,
                        "label_k562": 0,
                        "label_hepg2": 0,
                        "category": "negative",
                    }
                )
                total_collected += 1

        records = []
        for recs in collected_per_bin.values():
            records.extend(recs)
        return records
