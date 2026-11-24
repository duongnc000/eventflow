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


#seqs = [["TEMP", "COOL", "REC"],     # machine 1
#        ["TEMP", "COOL"]]            # machine 2
#ngram_counts(seqs, 2)
#-> {("TEMP","COOL"): 2, ("COOL","REC"): 1}
def ngram_counts(seqs, n):
    """How often each n-gram appears, counted inside each sequence (never across two sequences)."""
    counts = Counter()
    for seq in seqs:
        counts.update(ngrams(seq, n))
    return counts


def js_divergence(counts_a, counts_b): #real tally (counts_a): TEMP 2, COOL 2, fake tally (counts_b): TEMP 3, REC 1
    """Jensen-Shannon divergence of two count tables, in bits: 0 = identical, 1 = nothing shared."""
    keys = list(set(counts_a) | set(counts_b)) # => combine both then remove dup => [TEMP, COOL, REC]
    p = np.array([counts_a.get(k, 0) for k in keys], dtype=float) #p:2,2,0 
    q = np.array([counts_b.get(k, 0) for k in keys], dtype=float) #q:3,0,1
    p, q = p / p.sum(), q / q.sum() #turn to share p:0.5,0.5,0 q:0.75,0,0.25
    m = (p + q) / 2  #make the middle m: 0.625,0.25, 0.125

    def kl(x):                      # "distance" of the probablity distrubtion to the middle between the two prob dis p and q, smaller kl =  more similar to the middle
        nz = x > 0
        return np.sum(x[nz] * np.log2(x[nz] / m[nz]))

    return float((kl(p) + kl(q)) / 2) #average kl, smaller aver kl mean p and q similar, 0 identical, 1 very dif


def event_js(real, fake): # compute JS div between real and fake sequences wihout caring about order
    """1. MIX: compares how often each event appears. Ignores order."""
    return js_divergence(ngram_counts(real, 1), ngram_counts(fake, 1))


def ngram_js(real, fake, n=2): #  compute JS div between real and fake sequences but care about order
    """2. ORDER: compares how often each run of n events appears (n=2: 'A then B')."""
    return js_divergence(ngram_counts(real, n), ngram_counts(fake, n))


def unique_ngram_ratio(seqs, n=3): # ratio differnt window/ total window (seqs = [["A", "B", "C", "A", "B", "C", "A", "B", "C"]] => different ones: 3 (ABC, BCA, CAB), total: 7(ABC, BCA, CAB, ABC, BCA, CAB, ABC)) => 3/7
    """3a. VARIETY: distinct n-grams / all n-grams. Low = the same patterns over and over."""
    counts = ngram_counts(seqs, n)
    total = sum(counts.values())
    return len(counts) / total if total else float("nan") # ratio = 1 no repeat, repeat close to 0 : repeat alot


def duplicate_rate(seqs): # count repeating of the whole sequence
    """3b. VARIETY: share of sequences that are an exact copy of another one in the same set."""
    seen, dupes = set(), 0
    for seq in seqs:
        key = tuple(seq)
        dupes += key in seen
        seen.add(key)
    return dupes / len(seqs) #ratio dupes /total seqs


def copy_rate(fake, train_seqs): # check if the fake/gen seqence is just taking the window from training seq to gen
    """3c. MEMORISATION: share of fake sequences found word for word somewhere in the training data."""
    windows = set()
    for length in {len(seq) for seq in fake}: # check for every length of the fake seq
        for seq in train_seqs: #for each seq in train
            windows.update(hash(w) for w in ngrams(seq, length)) # hash each window of seq in train match length with fake seq
    return sum(hash(tuple(seq)) in windows for seq in fake) / len(fake) # ratio of fake seqs that exactly match some training window


def nearest_neighbour_similarity(seqs, train_seqs): 
#train_seqs = [["A","B","C","D"], ["C","D","E"]]
#seqs       = [["B","C","D"],          # fake 1
              #["B","A","D"],          # fake 2
              #["E","A","B"]]
    """3d. MEMORISATION: for each sequence, the share of positions it has in common with its
    closest training window of the same length (1.0 = an exact copy). Returns the average.

    Compare fake against real held-out sequences: fake much higher than real = copying.
    Takes about 30-60 s for 1,300 sequences of 20 events against ~4,000 training sequences.
    """
    vocab = {e: i for i, e in enumerate(sorted({e for s in train_seqs for e in s}))} #give vocab {A: 0, B: 1, C: 2, D: 3, E: 4}.
    best = []
    for length in sorted({len(s) for s in seqs}):
        # every run of `length` events in the training data, one per row
        #diary 1  A B C D → [0, 1, 2, 3] → [0, 1, 2][1, 2, 3]
        #diary 2  C D E   → [2, 3, 4]    → [2, 3, 4] => window = [0, 1, 2]
        #                                                        [1, 2, 3]
        #                                                        [2, 3, 4]   
        windows = np.concatenate([
            np.lib.stride_tricks.sliding_window_view(np.array([vocab[e] for e in s], np.int8), length) # convert the trainign seq into numbers using vocab then slide the training seq by the lenght of each fake seq
            for s in train_seqs if len(s) >= length
        ])

        for s in seqs:
            if len(s) == length:
                codes = np.array([vocab.get(e, -1) for e in s], np.int8)   # convert code= [1 2 3][1 0 3][4 0 1]to num -1 = never seen in training 
                matches = (windows == codes).sum(axis=1)                    # matching each code seq with the window(sum the matching number)
                best.append(matches.max() / length) # take the max ratio for each code compare to window best = [1.0, 0.667, 0.0]
    return float(np.mean(best)) #averge 0.5556


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