"""Checks for generated event sequences (project brief, section 4.4).

Every function takes a list of sequences, and each sequence is a list of events, e.g.
    real = [["TEMP_WARNING", "COOLING_ACTIVATED", "RECOVERY"], ["LOAD_INCREASE", ...], ...]
Event names or class codes (0..13) both work.

Three kinds of check:
  1. MIX      are the events equally common in real and fake?          event_js
  2. ORDER    do events follow each other the same way?                ngram_js (n=2, n=3)
  3. VARIETY  does the generator repeat itself or copy the training?   unique_ngram_ratio,
                                                                       duplicate_rate, copy_rate,
                                                                       nearest_neighbour_similarity
Compare real and fake sets of the SAME size and length: the variety numbers depend on both.
"""

from collections import Counter

import numpy as np


def ngrams(seq, n): #cuts sequence  into every run of n in a row
    """All runs of n events in a row: ngrams([A, B, C], 2) -> [(A, B), (B, C)]."""
    seq = list(seq)
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def ngram_counts(seqs, n):
    """How often each n-gram appears, counted inside each sequence (never across two sequences)."""
    counts = Counter()
    for seq in seqs:
        counts.update(ngrams(seq, n))
    return counts


def js_divergence(counts_a, counts_b):
    """Jensen-Shannon divergence of two count tables, in bits: 0 = identical, 1 = nothing shared."""
    keys = list(set(counts_a) | set(counts_b))
    p = np.array([counts_a.get(k, 0) for k in keys], dtype=float)
    q = np.array([counts_b.get(k, 0) for k in keys], dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(x):                      # KL(x || m), skipping entries where x is 0
        nz = x > 0
        return np.sum(x[nz] * np.log2(x[nz] / m[nz]))

    return float((kl(p) + kl(q)) / 2)


def event_js(real, fake):
    """1. MIX: compares how often each event appears. Ignores order."""
    return js_divergence(ngram_counts(real, 1), ngram_counts(fake, 1))


def ngram_js(real, fake, n=2):
    """2. ORDER: compares how often each run of n events appears (n=2: 'A then B')."""
    return js_divergence(ngram_counts(real, n), ngram_counts(fake, n))


def unique_ngram_ratio(seqs, n=3):
    """3a. VARIETY: distinct n-grams / all n-grams. Low = the same patterns over and over."""
    counts = ngram_counts(seqs, n)
    total = sum(counts.values())
    return len(counts) / total if total else float("nan")


def duplicate_rate(seqs):
    """3b. VARIETY: share of sequences that are an exact copy of another one in the same set."""
    seen, dupes = set(), 0
    for seq in seqs:
        key = tuple(seq)
        dupes += key in seen
        seen.add(key)
    return dupes / len(seqs)


def copy_rate(fake, train_seqs):
    """3c. MEMORISATION: share of fake sequences found word for word somewhere in the training data."""
    windows = set()
    for length in {len(seq) for seq in fake}:
        for seq in train_seqs:
            windows.update(hash(w) for w in ngrams(seq, length))
    return sum(hash(tuple(seq)) in windows for seq in fake) / len(fake)


def nearest_neighbour_similarity(seqs, train_seqs):
    """3d. MEMORISATION: for each sequence, the share of positions it has in common with its
    closest training window of the same length (1.0 = an exact copy). Returns the average.

    Compare fake against real held-out sequences: fake much higher than real = copying.
    Takes about 30-60 s for 1,300 sequences of 20 events against ~4,000 training sequences.
    """
    vocab = {e: i for i, e in enumerate(sorted({e for s in train_seqs for e in s}))}
    best = []
    for length in sorted({len(s) for s in seqs}):
        # every run of `length` events in the training data, one per row
        windows = np.concatenate([
            np.lib.stride_tricks.sliding_window_view(np.array([vocab[e] for e in s], np.int8), length)
            for s in train_seqs if len(s) >= length
        ])
        for s in seqs:
            if len(s) == length:
                codes = np.array([vocab.get(e, -1) for e in s], np.int8)   # -1 = never seen in training
                matches = (windows == codes).sum(axis=1)                    # matching positions per window
                best.append(matches.max() / length)
    return float(np.mean(best))


def summary(real, fake, train_seqs=None):
    """All checks as one dict (one row of the report table). Real and fake = same prefixes."""
    row = {
        "event_js": event_js(real, fake),
        "bigram_js": ngram_js(real, fake, 2),
        "trigram_js": ngram_js(real, fake, 3),
        "unique_3gram_real": unique_ngram_ratio(real, 3),
        "unique_3gram_fake": unique_ngram_ratio(fake, 3),
        "duplicate_rate_real": duplicate_rate(real),
        "duplicate_rate_fake": duplicate_rate(fake),
    }
    if train_seqs is not None:
        row["copy_rate_real"] = copy_rate(real, train_seqs)
        row["copy_rate_fake"] = copy_rate(fake, train_seqs)
        row["nn_similarity_real"] = nearest_neighbour_similarity(real, train_seqs)
        row["nn_similarity_fake"] = nearest_neighbour_similarity(fake, train_seqs)
    return row