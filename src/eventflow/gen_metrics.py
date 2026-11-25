"""Checks for generated event sequences (project brief, section 4.4).

Every function takes a list of sequences, and each sequence is a list of events, e.g.
    real = [["TEMP_WARNING", "COOLING_ACTIVATED", "RECOVERY"], ["LOAD_INCREASE", ...], ...]
Events can be names or class codes (0..13), but every set passed to one call must use the SAME
format: a name never equals a code, so mixing them would silently report "nothing in common".

Three kinds of check:
  1. MIX      are the events equally common in real and fake?          event_js
  2. ORDER    do events follow each other the same way?                ngram_js (n=2, n=3)
  3. VARIETY  does the generator repeat itself or copy the training?   unique_ngram_ratio,
                                                                       duplicate_rate, copy_rate,
                                                                       nearest_neighbour_similarity
Compare real and fake sets of the SAME size and length: the variety numbers depend on both.
Empty inputs give NaN.
"""

from collections import Counter

import numpy as np


def _check_same_format(*sets):
    """Raise if one set uses event names and another uses class codes."""
    kinds = set()
    for seqs in sets:
        for seq in seqs:
            if len(seq):
                kinds.add(isinstance(seq[0], str))
                break
    if len(kinds) > 1:
        raise ValueError("mixed event formats: use event names everywhere or class codes everywhere")


def ngrams(seq, n):
    """All runs of n events in a row: ngrams([A, B, C], 2) -> [(A, B), (B, C)]."""
    seq = list(seq)
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def ngram_counts(seqs, n):
    """How often each n-gram appears, counted inside each sequence (never across two sequences).

    ngram_counts([["TEMP", "COOL", "REC"], ["TEMP", "COOL"]], 2)
    -> {("TEMP", "COOL"): 2, ("COOL", "REC"): 1}
    """
    counts = Counter()
    for seq in seqs:
        counts.update(ngrams(seq, n))
    return counts


def js_divergence(counts_a, counts_b):
    """Jensen-Shannon divergence of two count tables, in bits: 0 = identical, 1 = nothing shared.

    Example:  a = {TEMP: 2, COOL: 2}         -> p = [0.5,  0.5, 0   ]  (over TEMP, COOL, REC)
              b = {TEMP: 3, REC: 1}          -> q = [0.75, 0,   0.25]
              middle m = (p + q) / 2              = [0.625, 0.25, 0.125]
    JS = average of KL(p || m) and KL(q || m): how far each one is from the middle.
    """
    keys = list(set(counts_a) | set(counts_b))
    p = np.array([counts_a.get(k, 0) for k in keys], dtype=float)
    q = np.array([counts_b.get(k, 0) for k in keys], dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(x):
        # KL divergence of x from the middle m; terms with x = 0 contribute nothing
        nz = x > 0
        return np.sum(x[nz] * np.log2(x[nz] / m[nz]))

    return float((kl(p) + kl(q)) / 2)


def event_js(real, fake):
    """1. MIX: compares how often each event appears. Ignores order."""
    _check_same_format(real, fake)
    return js_divergence(ngram_counts(real, 1), ngram_counts(fake, 1))


def ngram_js(real, fake, n=2):
    """2. ORDER: compares how often each run of n events appears (n=2: 'A then B')."""
    _check_same_format(real, fake)
    return js_divergence(ngram_counts(real, n), ngram_counts(fake, n))


def unique_ngram_ratio(seqs, n=3):
    """3a. VARIETY: distinct n-grams / all n-grams. Low = the same patterns over and over.

    e.g. [[A, B, C, A, B, C, A, B, C]] has 7 trigrams but only 3 distinct (ABC, BCA, CAB) -> 3/7.
    """
    counts = ngram_counts(seqs, n)
    total = sum(counts.values())
    return len(counts) / total if total else float("nan")


def duplicate_rate(seqs):
    """3b. VARIETY: share of sequences that are an exact copy of an earlier one in the same set."""
    if len(seqs) == 0:
        return float("nan")
    seen, dupes = set(), 0
    for seq in seqs:
        key = tuple(seq)
        dupes += key in seen
        seen.add(key)
    return dupes / len(seqs)


def copy_rate(fake, train_seqs):
    """3c. MEMORISATION: share of fake sequences found word for word somewhere in the training data.

    A fake sequence of length L counts as a copy if it equals any run of L events inside one
    training sequence.
    """
    if len(fake) == 0:
        return float("nan")
    _check_same_format(fake, train_seqs)
    # Every training window with the same length as some fake sequence (stored as hashes to save memory)
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

    Example:  train = [[A, B, C, D], [C, D, E]]  -> windows of length 3: ABC, BCD, CDE
              seqs  = [[B, C, D], [B, A, D], [E, A, B]]
              best  = [3/3 (BCD), 2/3 (BAD vs BCD), 0/3]  -> average 0.556
    """
    if len(seqs) == 0:
        return float("nan")
    _check_same_format(seqs, train_seqs)
    # Events -> small integers, so the comparison below runs on numpy arrays
    vocab = {e: i for i, e in enumerate(sorted({e for s in train_seqs for e in s}))}
    best = []
    for length in sorted({len(s) for s in seqs}):
        # Every run of `length` events in the training data, one per row
        windows = [
            np.lib.stride_tricks.sliding_window_view(np.array([vocab[e] for e in s], np.int8), length)
            for s in train_seqs if len(s) >= length
        ]
        if not windows:
            raise ValueError(f"no training sequence has {length} events, so sequences of that "
                             "length have no training window to compare with")
        windows = np.concatenate(windows)

        for s in seqs:
            if len(s) == length:
                # -1 = an event never seen in training (matches nothing)
                codes = np.array([vocab.get(e, -1) for e in s], np.int8)
                matches = (windows == codes).sum(axis=1)   # positions in common with each window
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
