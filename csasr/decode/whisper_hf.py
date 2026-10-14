"""Batched Whisper decoding with explicit language prefixes (proposal §4.1).

Language modes:
  zh / en   force <|zh|> or <|en|>
  both      force <|zh|><|en|> (Peng et al. 2023)
  free      the model predicts the language token itself, restricted to {zh, en}
            (this is also how fine-tuned models are run: they were trained to emit
            the main-language token)
Output is capped at 128 new tokens.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (BitsAndBytesConfig, GenerationMixin, WhisperFeatureExtractor,
                          WhisperForConditionalGeneration, WhisperTokenizer)

from ..data.manifest import load_audio

MAX_NEW_TOKENS = 128


def hub_name(size_or_path: str) -> str:
    return f"openai/whisper-{size_or_path}" if size_or_path in ("tiny", "base", "small", "medium") else size_or_path


def load_model(size_or_path: str, precision: str = "fp16", device="cuda", skip_modules=None):
    """precision: fp16 | bf16 | fp32 | int8_bnb | nf4_bnb. Only nn.Linear layers are
    quantized by bitsandbytes; convs, embeddings, layer norms stay 16-bit and the
    output layer (proj_out) is skipped explicitly."""
    name = hub_name(size_or_path)
    kw = dict(torch_dtype=torch.float16)
    skip = ["proj_out"] + list(skip_modules or [])
    if precision == "int8_bnb":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        kw["device_map"] = {"": device}
    elif precision == "nf4_bnb":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False, llm_int8_skip_modules=skip)
        kw["device_map"] = {"": device}
    elif precision == "bf16":
        kw["torch_dtype"] = torch.bfloat16
    elif precision == "fp32":
        kw["torch_dtype"] = torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(name, **kw)
    if "quantization_config" not in kw:
        model.to(device)
    model.eval()
    gc = model.generation_config
    gc.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    return model


def load_processor(size_or_path: str):
    name = hub_name(size_or_path)
    try:
        tok = WhisperTokenizer.from_pretrained(name)
        fe = WhisperFeatureExtractor.from_pretrained(name)
    except OSError:  # fine-tuned dirs without processor files
        tok = fe = None
    return tok, fe


def quantized_modules(model) -> list[str]:
    import bitsandbytes as bnb
    return [n for n, m in model.named_modules() if isinstance(m, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit))]


class _AudioSet(Dataset):
    def __init__(self, rows, fe):
        self.rows, self.fe = rows, fe

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        wav = load_audio(self.rows[i]["audio"])
        feats = self.fe(wav, sampling_rate=16000, return_tensors="np").input_features[0]
        return i, feats


def _collate(b):
    idx = [x[0] for x in b]
    return idx, torch.from_numpy(np.stack([x[1] for x in b]))


class Decoder:
    def __init__(self, model, tok, fe, device="cuda"):
        if hasattr(model, "get_base_model"):  # PEFT: LoRA layers are injected into the base model
            model = model.get_base_model()
        self.model, self.tok, self.fe, self.device = model, tok, fe, device
        t = tok.convert_tokens_to_ids
        self.sot, self.trans, self.notime = t("<|startoftranscript|>"), t("<|transcribe|>"), t("<|notimestamps|>")
        self.lang = {"zh": t("<|zh|>"), "en": t("<|en|>")}
        self.eot = tok.eos_token_id
        self.dtype = next(p for p in model.parameters() if p.is_floating_point()).dtype
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            self.dtype = torch.float16

    def prefix(self, langs: list[str]) -> torch.Tensor:
        rows = []
        for l in langs:
            ls = [self.lang["zh"], self.lang["en"]] if l == "both" else [self.lang[l]]
            rows.append([self.sot] + ls + [self.trans, self.notime])
        return torch.tensor(rows, device=self.device)

    @torch.no_grad()
    def predict_lang(self, enc_out) -> list[str]:
        """One decoder step after <|sot|>, argmax over {zh, en}."""
        B = enc_out.last_hidden_state.shape[0]
        dec = torch.full((B, 1), self.sot, device=self.device)
        logits = self.model(encoder_outputs=enc_out, decoder_input_ids=dec).logits[:, -1]
        ids = torch.tensor([self.lang["zh"], self.lang["en"]], device=self.device)
        pick = logits[:, ids].argmax(-1).tolist()
        return ["zh" if p == 0 else "en" for p in pick]

    @torch.no_grad()
    def generate(self, feats, lang_mode: str, num_beams=1, num_return_sequences=1, langs=None,
                 output_scores=False):
        feats = feats.to(self.device, dtype=self.dtype)
        enc = self.model.get_encoder()(feats)
        if langs is None:
            langs = self.predict_lang(enc) if lang_mode == "free" else [lang_mode] * feats.shape[0]
        dec = self.prefix(langs)
        out = GenerationMixin.generate(
            self.model, encoder_outputs=enc, decoder_input_ids=dec, max_new_tokens=MAX_NEW_TOKENS,
            num_beams=num_beams, num_return_sequences=num_return_sequences, do_sample=False,
            eos_token_id=self.eot, pad_token_id=self.eot, return_dict_in_generate=True,
            output_scores=output_scores, length_penalty=1.0,
            suppress_tokens=self.model.generation_config.suppress_tokens)
        seqs = out.sequences[:, dec.shape[1]:]
        texts = self.tok.batch_decode(seqs, skip_special_tokens=True)
        res = dict(texts=[t.strip() for t in texts], langs=langs, seqs=seqs)
        if output_scores and num_beams > 1:
            res["seq_scores"] = out.sequences_scores.tolist()
        return res

    def transcribe(self, rows, lang_mode="free", batch_size=32, num_workers=8, num_beams=1):
        order = sorted(range(len(rows)), key=lambda i: -rows[i].get("duration", 0))
        ds = _AudioSet([rows[i] for i in order], self.fe)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate)
        out = [None] * len(rows)
        for idx, feats in dl:
            r = self.generate(feats, lang_mode, num_beams=num_beams)
            for k, i in enumerate(idx):
                out[order[i]] = (r["texts"][k], r["langs"][k])
        return out
