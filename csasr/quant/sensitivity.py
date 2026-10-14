"""Layer-group sensitivity scan (proposal §4.4, Appendix A6).

S_g = 1/|T_emb| sum_{t in T_emb} KL(p16(.|y<t,x) || p_g(.|y<t,x)), where only layer group g
(one encoder or decoder block) is quantized and T_emb are the embedded-unit token positions
of the validation set under teacher forcing.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..decode.whisper_hf import load_model
from ..train.data import ASRDataset, collate


def block_names(model) -> list[str]:
    n_enc = model.config.encoder_layers
    n_dec = model.config.decoder_layers
    return [f"model.encoder.layers.{i}" for i in range(n_enc)] + [f"model.decoder.layers.{i}" for i in range(n_dec)]


def skip_all_but(blocks: list[str], keep_quantized: list[str]) -> list[str]:
    """llm_int8_skip_modules list that leaves only `keep_quantized` blocks quantized."""
    return [b for b in blocks if b not in keep_quantized]


class EmbeddedTokenBatches:
    """Teacher-forced batches of validation clips, with the embedded-token mask."""

    def __init__(self, rows, tok, fe, batch=32, num_workers=8):
        ds = ASRDataset(rows, tok, fe)
        self.batches = list(DataLoader(ds, batch_size=batch, num_workers=num_workers,
                                       collate_fn=lambda b: collate(b, tok.eos_token_id)))

    def __iter__(self):
        return iter(self.batches)


@torch.no_grad()
def logprobs_at_embedded(model, batches, dtype=torch.float16):
    """List (per batch) of log-prob tensors [n_emb_positions, vocab] (kept on CPU, fp16)."""
    out = []
    for b in batches:
        logits = model(input_features=b["input_features"].cuda().to(dtype),
                       decoder_input_ids=b["decoder_input_ids"].cuda()).logits
        mask = (b["roles"] == 2).cuda()
        out.append(F.log_softmax(logits[mask].float(), -1).half().cpu())
    return out


@torch.no_grad()
def kl_vs_reference(model, batches, ref_lp, dtype=torch.float16) -> float:
    tot, n = 0.0, 0
    for b, lp16 in zip(batches, ref_lp):
        logits = model(input_features=b["input_features"].cuda().to(dtype),
                       decoder_input_ids=b["decoder_input_ids"].cuda()).logits
        mask = (b["roles"] == 2).cuda()
        lq = F.log_softmax(logits[mask].float(), -1)
        lp = lp16.cuda().float()
        tot += (lp.exp() * (lp - lq)).sum().item()
        n += lp.shape[0]
    return tot / max(n, 1)


def scan(model_path: str, rows, tok, fe, precision="nf4_bnb", log=print) -> list[tuple[str, float]]:
    m16 = load_model(model_path, "fp16")
    blocks = block_names(m16)
    batches = EmbeddedTokenBatches(rows, tok, fe)
    ref = logprobs_at_embedded(m16, batches)
    del m16
    torch.cuda.empty_cache()
    res = []
    for g in blocks:
        mq = load_model(model_path, precision, skip_modules=skip_all_but(blocks, [g]))
        s = kl_vs_reference(mq, batches, ref)
        log(f"{g}: KL={s:.5f}")
        res.append((g, s))
        del mq
        torch.cuda.empty_cache()
    full = load_model(model_path, precision)
    res.append(("ALL", kl_vs_reference(full, batches, ref)))
    del full
    return res
