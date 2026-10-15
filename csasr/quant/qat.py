"""Quantization-aware fine-tuning, QLoRA style (proposal §4.4).

Freeze the NF4-quantized fine-tuned model and train rank-16 LoRA adapters on top with the
same loss. Adapters stay 16-bit at inference. For CPU deployment the adapters are merged
into the 16-bit weights and the model is re-quantized (merge_adapter()).
"""
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, WhisperForConditionalGeneration

from ..data.manifest import path
from ..decode.whisper_hf import hub_name

TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def _src(p):
    return path(p) if os.path.isdir(path(p)) else hub_name(p)


def build_qlora_model(model_path: str, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           llm_int8_skip_modules=["proj_out"])
    model = WhisperForConditionalGeneration.from_pretrained(_src(model_path), quantization_config=q,
                                                            device_map={"": "cuda"}, torch_dtype=torch.bfloat16)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS, bias="none")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def load_qlora_for_eval(base_path: str, adapter_dir: str):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                           llm_int8_skip_modules=["proj_out"])
    base = WhisperForConditionalGeneration.from_pretrained(_src(base_path), quantization_config=q,
                                                           device_map={"": "cuda"}, torch_dtype=torch.float16)
    m = PeftModel.from_pretrained(base, path(adapter_dir)).eval()
    m.generation_config.forced_decoder_ids = None
    return m


def merge_adapter(base_path: str, adapter_dir: str, out_dir: str):
    """fp16 base + LoRA -> merged fp16 checkpoint (then re-quantize with any backend)."""
    base = WhisperForConditionalGeneration.from_pretrained(_src(base_path), torch_dtype=torch.float16)
    m = PeftModel.from_pretrained(base, path(adapter_dir)).merge_and_unload()
    m.generation_config.forced_decoder_ids = None
    m.save_pretrained(path(out_dir), safe_serialization=True)
    # exporters (CTranslate2, whisper.cpp) need the tokenizer and feature extractor next to the weights
    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    WhisperTokenizer.from_pretrained("openai/whisper-small").save_pretrained(path(out_dir))
    WhisperFeatureExtractor.from_pretrained("openai/whisper-small").save_pretrained(path(out_dir))
    return path(out_dir)
