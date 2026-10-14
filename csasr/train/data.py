"""Training data: features, decoder targets and per-token role labels (proposal §4.1, A1).

Target sequence: <|sot|> <|main-lang|> <|transcribe|> <|notimestamps|> text <|eot|>.
The language token is predicted by the model (it is trained like a text token);
task / no-timestamps tokens are forced at inference and get no loss.

Token roles: each text token's byte span in the UTF-8 target is matched against the
byte spans of reference units; a token covering units with different roles takes
the larger weight (E > Ms > others).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.manifest import load_audio
from ..text.units import label, split_units, training_target

ROLE_CODE = {"E": 2, "Ms": 1}  # 0 = other (Mi, neu, bal); -1 = no loss; 3 = language token
MAX_LABEL_TOKENS = 128 + 4


class TokenSpans:
    """Byte spans of byte-level BPE tokens (exact even when a Han char spans 2 tokens)."""

    def __init__(self, tok):
        self.tok = tok
        self.byte_decoder = tok.byte_decoder

    def spans(self, ids):
        out, pos = [], 0
        for t in self.tok.convert_ids_to_tokens(ids):
            n = len(t)  # each char of a byte-level token is one byte
            out.append((pos, pos + n))
            pos += n
        return out


def unit_byte_spans(target: str, units: list[str]):
    spans, pos = [], 0
    b = target.encode("utf-8")
    for u in units:
        ub = u.encode("utf-8")
        k = b.find(ub, pos)
        if k < 0:
            raise ValueError(f"unit {u!r} not found in {target!r}")
        spans.append((k, k + len(ub)))
        pos = k + len(ub)
    return spans


def token_roles(text: str, ids, spanner: TokenSpans) -> list[int]:
    """text: the exact string that was tokenized into ids (may start with a space)."""
    lab = label(text)
    tok_spans = spanner.spans(ids)
    u_spans = unit_byte_spans(text, lab.units)
    codes = []
    j = 0
    for s, e in tok_spans:
        best = 0
        while j < len(u_spans) and u_spans[j][1] <= s:
            j += 1
        k = j
        while k < len(u_spans) and u_spans[k][0] < e:
            best = max(best, ROLE_CODE.get(lab.roles[k], 0))
            k += 1
        codes.append(best)
    return codes


class ASRDataset(Dataset):
    def __init__(self, rows, tok, fe, with_roles=True):
        self.rows, self.tok, self.fe = rows, tok, fe
        self.with_roles = with_roles
        t = tok.convert_tokens_to_ids
        self.sot, self.trans, self.notime = t("<|startoftranscript|>"), t("<|transcribe|>"), t("<|notimestamps|>")
        self.lang = {"zh": t("<|zh|>"), "en": t("<|en|>")}
        self.eot = tok.eos_token_id
        self.spanner = TokenSpans(tok)

    def __len__(self):
        return len(self.rows)

    def main_lang(self, r):
        m = r.get("main")
        if m in ("zh", "en"):
            return m
        units = split_units(r["ref"])  # balanced: language of the first non-neutral unit
        for u in units:
            if u.isascii() and not u.isdigit():
                return "en"
            if not u.isascii():
                return "zh"
        return "en"

    def encode(self, r):
        target = r.get("target") or training_target(r["ref_raw"])
        # Whisper writes Latin-initial text with a leading space (" hello"), Chinese without
        text = " " + target if target and target[0].isascii() else target
        ids = self.tok(text, add_special_tokens=False).input_ids
        roles = token_roles(text, ids, self.spanner) if self.with_roles else [0] * len(ids)
        ids, roles = ids[:MAX_LABEL_TOKENS - 5], roles[:MAX_LABEL_TOKENS - 5]
        seq = [self.sot, self.lang[self.main_lang(r)], self.trans, self.notime] + ids + [self.eot]
        # role code per *target* position (predicting seq[t+1] from seq[:t+1])
        tgt_roles = [3, -1, -1] + roles + [0]
        return seq, tgt_roles

    def __getitem__(self, i):
        r = self.rows[i]
        wav = load_audio(r["audio"])
        feats = self.fe(wav, sampling_rate=16000, return_tensors="np").input_features[0]
        seq, roles = self.encode(r)
        return dict(feats=feats, seq=seq, roles=roles, idx=i)


def collate(batch, pad_id):
    T = max(len(b["seq"]) for b in batch)
    B = len(batch)
    dec = torch.full((B, T - 1), pad_id, dtype=torch.long)
    tgt = torch.full((B, T - 1), -100, dtype=torch.long)
    roles = torch.full((B, T - 1), -1, dtype=torch.long)
    for k, b in enumerate(batch):
        s = b["seq"]
        n = len(s) - 1
        dec[k, :n] = torch.tensor(s[:-1])
        tgt[k, :n] = torch.tensor(s[1:])
        roles[k, :n] = torch.tensor(b["roles"])
    tgt[roles == -1] = -100
    return dict(input_features=torch.from_numpy(np.stack([b["feats"] for b in batch])),
                decoder_input_ids=dec, labels=tgt, roles=roles, idx=[b["idx"] for b in batch])
