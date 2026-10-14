"""Two-pass scoring (proposal §4.2, Appendix A7).

Pass 1: unit-level minimum edit distance -> S, D, I, N, MER (official, comparable).
Pass 2: group consecutive errors between two correct units into error blocks and
charge every edit to exactly one reference unit, so that role rates add back to MER.
"""
from dataclasses import dataclass, field

import numpy as np

# op codes
C, S, D, I = "C", "S", "D", "I"


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein alignment. Ties: prefer substitution, then deletion, then insertion,
    so a 1-for-1 swap is never split into insertion + deletion."""
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        r = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (r != hyp[j - 1])
            dele = prev[j] + 1
            ins = row[j - 1] + 1
            row[j] = min(sub, dele, ins)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((C if ref[i - 1] == hyp[j - 1] else S, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops.append((D, i - 1, None))
            i -= 1
        else:
            ops.append((I, None, j - 1))
            j -= 1
    ops.reverse()
    return ops


@dataclass
class Block:
    ops: list                 # the non-C ops of this block, in order
    ref_idx: list[int]        # reference units in the block (S and D)
    hyp_idx: list[int]        # hypothesis units in the block (S and I)
    n_ins: int
    left: int | None          # reference index of the correct unit before the block
    right: int | None         # reference index of the correct unit after the block
    confusion: bool = False   # language-confusion (translation) block, set by confusion.py
    sim: float = 0.0          # LaBSE similarity ref-side vs hyp-side (if computed)


@dataclass
class Scored:
    n_ref: int
    S: int
    D: int
    I: int
    ops: list
    blocks: list[Block]
    kappa: np.ndarray                 # error charged to each reference unit (A7)
    unit_err: np.ndarray              # 1 if the reference unit is substituted/deleted
    unit_correct: np.ndarray          # 1 if the reference unit is aligned as correct
    orphan: float = 0.0               # insertions with no reference unit to charge (empty ref)
    extra: dict = field(default_factory=dict)

    @property
    def errors(self):
        return self.S + self.D + self.I


def blocks_from_ops(ops) -> list[Block]:
    blocks, cur, last_c = [], [], None
    for op, ri, hj in ops:
        if op == C:
            if cur:
                blocks.append(_mk_block(cur, last_c, ri))
                cur = []
            last_c = ri
        else:
            cur.append((op, ri, hj))
    if cur:
        blocks.append(_mk_block(cur, last_c, None))
    return blocks


def _mk_block(cur, left, right) -> Block:
    ref_idx = [ri for op, ri, _ in cur if op in (S, D)]
    hyp_idx = [hj for op, _, hj in cur if op in (S, I)]
    return Block(cur, ref_idx, hyp_idx, sum(op == I for op, _, _ in cur), left, right)


def score(ref: list[str], hyp: list[str]) -> Scored:
    ops = align(ref, hyp)
    n = len(ref)
    S_ = sum(o == S for o, _, _ in ops)
    D_ = sum(o == D for o, _, _ in ops)
    I_ = sum(o == I for o, _, _ in ops)
    kappa = np.zeros(n)
    unit_err = np.zeros(n)
    unit_correct = np.zeros(n)
    for o, ri, _ in ops:
        if o in (S, D):
            unit_err[ri] = 1
        elif o == C:
            unit_correct[ri] = 1
    blocks = blocks_from_ops(ops)
    orphan = 0.0
    for b in blocks:
        if b.ref_idx:
            share = b.n_ins / len(b.ref_idx)
            for ri in b.ref_idx:
                kappa[ri] += 1 + share
        else:  # pure insertion block: share between correct neighbours
            nb = [x for x in (b.left, b.right) if x is not None]
            if nb:
                for x in nb:
                    kappa[x] += b.n_ins / len(nb)
            else:
                orphan += b.n_ins
    assert abs(kappa.sum() + orphan - (S_ + D_ + I_)) < 1e-6
    return Scored(n, S_, D_, I_, ops, blocks, kappa, unit_err, unit_correct, orphan)
