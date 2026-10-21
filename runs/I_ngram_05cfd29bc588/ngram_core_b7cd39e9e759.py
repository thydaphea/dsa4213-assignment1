
"""DSA4213 sparse interpolated count LM. Only NumPy and Python stdlib needed."""
from collections import Counter
import math
import numpy as np

class InterpolatedNGram:
    def __init__(self, vocab_size, predicted_ids, pad=0, bos=2):
        self.vocab_size = int(vocab_size)
        self.allowed = np.asarray(predicted_ids, dtype=np.int64)
        self.pad, self.bos = pad, bos
        self.K = len(self.allowed)
        self.unigrams = Counter()
        self.bigrams, self.trigrams = {}, {}
        self.bigram_totals, self.trigram_totals = Counter(), Counter()
        self.total = 0
        self.k, self.weights = None, None

    def fit(self, x, y):
        if self.total:
            raise RuntimeError('Fit a fresh instance; do not count the same corpus twice.')
        for inputs, targets in zip(x, y):
            for t, target in enumerate(targets):
                w = int(target)
                if w == self.pad:
                    continue
                h1 = int(inputs[t])
                self.unigrams[w] += 1
                self.bigrams.setdefault(h1, Counter())[w] += 1
                self.bigram_totals[h1] += 1
                # At t=0 the trigram component later reuses the bigram;
                # do not pretend that a second history token exists.
                if t >= 1:
                    h2 = (int(inputs[t-1]), h1)
                    self.trigrams.setdefault(h2, Counter())[w] += 1
                    self.trigram_totals[h2] += 1
                self.total += 1
        if not self.total:
            raise ValueError('Empty training targets.')
        return self

    def set_smoothing(self, k, weights):
        if k <= 0 or len(weights) != 3 or min(weights) < 0 or not np.isclose(sum(weights), 1):
            raise ValueError('Require k>0 and three nonnegative weights summing to 1.')
        self.k, self.weights = float(k), tuple(float(v) for v in weights)
        return self

    def counts_for_targets(self, x, y):
        """One row per valid target, in saved block-major order; no future inputs."""
        n = int(np.count_nonzero(y != self.pad))
        counts = np.empty((n, 3), dtype=np.float64)
        totals = np.empty_like(counts)
        at = 0
        for inputs, targets in zip(x, y):
            for t, target in enumerate(targets):
                w = int(target)
                if w == self.pad:
                    continue
                h1 = int(inputs[t])
                c1, d1 = self.unigrams.get(w, 0), self.total
                c2 = self.bigrams.get(h1, {}).get(w, 0)
                d2 = self.bigram_totals.get(h1, 0)
                if t:
                    h2 = (int(inputs[t-1]), h1)
                    c3 = self.trigrams.get(h2, {}).get(w, 0)
                    d3 = self.trigram_totals.get(h2, 0)
                else:
                    c3, d3 = c2, d2
                counts[at] = c1, c2, c3
                totals[at] = d1, d2, d3
                at += 1
        return counts, totals

    def score_counts(self, counts, totals, k=None, weights=None):
        k = self.k if k is None else k
        weights = self.weights if weights is None else weights
        probabilities = ((counts + k) / (totals + k * self.K)) @ np.asarray(weights)
        if not len(probabilities) or not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0):
            raise ValueError('Invalid probability or empty scored split.')
        nll = float(-np.log(probabilities).sum(dtype=np.float64))
        ce = nll / len(probabilities)
        return {'nll_sum': nll, 'token_count': len(probabilities), 'ce': ce,
                'ppl': math.exp(ce) if ce < math.log(np.finfo(float).max) else None}

    def distribution(self, history):
        """Full vocabulary vector, with zero PAD/BOS probability."""
        if self.k is None:
            raise RuntimeError('Select smoothing before prediction.')
        history = tuple(int(v) for v in history)
        tables = [self.unigrams]
        totals = [self.total]
        if history:
            h1 = history[-1]
            tables.append(self.bigrams.get(h1, {}))
            totals.append(self.bigram_totals.get(h1, 0))
        else:
            tables.append(tables[0]); totals.append(totals[0])
        if len(history) >= 2:
            h2 = history[-2:]
            tables.append(self.trigrams.get(h2, {}))
            totals.append(self.trigram_totals.get(h2, 0))
        else:
            tables.append(tables[1]); totals.append(totals[1])
        out = np.zeros(self.vocab_size, dtype=np.float64)
        for weight, table, total in zip(self.weights, tables, totals):
            denominator = total + self.k * self.K
            out[self.allowed] += weight * self.k / denominator
            if table:
                ids = np.fromiter(table.keys(), dtype=np.int64, count=len(table))
                values = np.fromiter(table.values(), dtype=np.float64, count=len(table))
                out[ids] += weight * values / denominator
        return out

    def entry_counts(self):
        return {'unigram_entries': len(self.unigrams),
                'bigram_entries': sum(map(len, self.bigrams.values())),
                'trigram_entries': sum(map(len, self.trigrams.values())),
                'bigram_contexts': len(self.bigrams), 'trigram_contexts': len(self.trigrams)}


def generate(model, prompt_ids, *, bos=2, eos=3, forbidden=(0, 1, 2),
             max_new_tokens=50, decoding='greedy', temperature=0.8, top_k=50,
             seed=123, stop_at_eos=True):
    if decoding not in ('greedy', 'sampling'):
        raise ValueError('Unknown decoder.')
    rng = np.random.default_rng(seed)
    prefix = [bos] + list(prompt_ids)
    generated = []
    for _ in range(max_new_tokens):
        probabilities = model.distribution(prefix[-2:])
        probabilities[list(forbidden)] = 0.0
        if decoding == 'greedy':
            next_id = int(np.argmax(probabilities))  # lowest ID breaks ties
        else:
            eligible = np.flatnonzero(probabilities > 0)
            # Stable descending sort: lower token ID wins exact ties.
            eligible = eligible[np.argsort(-probabilities[eligible], kind='stable')][:top_k]
            logits = np.log(probabilities[eligible]) / temperature
            weights = np.exp(logits - logits.max())
            weights /= weights.sum()
            next_id = int(rng.choice(eligible, p=weights))
        generated.append(next_id)
        prefix.append(next_id)
        if stop_at_eos and next_id == eos:
            break
    return {'token_ids': generated, 'generated_length': len(generated),
            'stop_reason': 'eos' if stop_at_eos and generated and generated[-1] == eos else 'length_cap',
            'eos_count': generated.count(eos)}
